import torch
import os
import sys

from student import *
from teacher import *
from data_utils import *
from torchdistill_utils import *
from utils import *
from torchdistill.models.registry import get_model

from torchdistill.core.forward_hook import ForwardHookManager

import yaml
from startup_config import set_random_seed
from menu import get_main_menu
from main import get_train_dev_dataloader

from utils import EarlyStopping
import logging
from tensorboardX import SummaryWriter
from tqdm import tqdm

import wandb
from datetime import timedelta
from wandb import AlertLevel
from contrast.supcontrastloss import SupConLoss
from engine.dot import DistillationOrientedTrainer
import eval_metrics_DF as em
from aasist.AASIST import *

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# Disable pydub logging
logging.getLogger('pydub.converter').setLevel(logging.CRITICAL)

# Set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
logger.info('Device: {}'.format(device))


def train_one_epoch(train_loader, model, criterion, optimizer, device, scaler, config, exp_lr_scheduler=None, use_amp: bool = True):
    running_loss = 0
    num_correct = 0.0
    model.train()
    num_total = 0.0
    iters = len(train_loader)
    pbar = tqdm(enumerate(train_loader), total=len(train_loader))
    for i, (batch_x, batch_y) in pbar:
        # Mixed precision training
        with torch.autocast(device_type=device, dtype=torch.float16, enabled=use_amp):
            batch_size = batch_x.size(0)
            num_total += batch_size
            batch_x = batch_x.to(device)
            batch_y = batch_y.view(-1).type(torch.int64).to(device)
            batch_out = model(batch_x)
            loss = criterion(batch_out, batch_y)
            running_loss += (loss.item() * batch_size)
        # Scaler
        optimizer.zero_grad()
        # Backward pass
        scaler.scale(loss).backward()

        # Update the weights
        scaler.step(optimizer)

        # Update the scaler
        scaler.update()

        _, batch_pred = batch_out.max(dim=1)
        # batch_y = batch_y.view(-1)
        num_correct += (batch_pred == batch_y).sum(dim=0).item()

        # Update LR
        if exp_lr_scheduler is not None:
            if config['learning_rate_scheduler']['name'] == 'CosineAnnealingWarmRestarts':
                exp_lr_scheduler.step(epoch + i / iters)
            elif config['learning_rate_scheduler']['name'] in ['ReduceLROnPlateau', 'MultiStepLR', 'StepLR']:
                # Update learning rate scheduler in validation so do nothing here
                pass
            else:
                exp_lr_scheduler.step()
    running_loss /= num_total
    accuracy = (num_correct / num_total) * 100
    return running_loss, accuracy


def validation(dev_loader, model, criterion, device):
    model.eval()
    num_total = 0.0
    num_correct = 0.0
    val_loss = 0
    with torch.no_grad():
        for batch_x, batch_y in tqdm(dev_loader):
            batch_size = batch_x.size(0)
            num_total += batch_size
            batch_x = batch_x.to(device)
            batch_y = batch_y.view(-1).type(torch.int64).to(device)
            batch_out = model(batch_x)
            loss = criterion(batch_out, batch_y)
            val_loss += (loss.item() * batch_size)
            _, batch_pred = batch_out.max(dim=1)
            num_correct += (batch_pred == batch_y).sum(dim=0).item()
    val_loss /= num_total
    accuracy = (num_correct / num_total) * 100

    return val_loss, accuracy


args = get_main_menu()
# Load configuration
with open(args.yaml, 'r') as f:
    logger.info('Load configuration file {}'.format(args.yaml))
    config = yaml.safe_load(f)


seed = config['train'].get('seed', 1234)
ce_weight = torch.FloatTensor(config['train'].get(
    'cross_entropy_loss_weight', [0.1, 0.9])).to(device)
student_resume_path = config['train'].get('student_resume', None)
sup_contrastive = config['train'].get('sup_contrastive', False)
is_learning_rate_scheduler = config['train'].get(
    'is_learning_rate_scheduler', False)
patience = config['train'].get('patience', 10)
use_amp = config['train'].get('amp', False)
learning_rate_scheduler_name = config['learning_rate_scheduler'].get(
    'name', None)
student_model_name = config['model']['student'].get(
    'name', 'Distil_W2V2BASE_AASISTL')
teacher_model_name = config['model']['teacher'].get('name', 'W2V2_TA')
model_path = config['model']['teacher'].get('pretrained_path', None)
student_model_path = config['train'].get('student_resume', None)
student_model_type = config['model']['student'].get(
    'type', 'ssl')
augment_mode = config["train"].get("augment_mode", "rawboost")
dataset = config["train"].get("dataset", "LA19")
dot = config["train"].get("dot", False)
mixup = config["train"].get("mixup", False)
restore = config["train"].get("restore", False)
teacher_dict = config["model"].get("teacher_multi", {})
copy_weights = config["train"].get("copy_weights", False)
is_teacher_parallel = config["model"]["teacher"].get("is_parallel", True)
freeze_layers = config["train"].get("freeze_layers", [])
encoder_layerdrop = config["model"]["student"].get("kwargs", {}).get(
    "encoder_layerdrop", 0.0)

custom_order_copy_weights = config["model"]["student"].get("kwargs", {}).get(
    "custom_order", [])
order = config["model"]["student"].get("kwargs", {}).get(
    "order", None)

wandb_project_name = config["train"].get("wandb_project_name", "torchdistill")
byot_kd_train = config["train"].get("byot_kd_train", False)
is_recon_loss = config['train'].get('is_recon_loss', False)
teacher_module_list = []

logger.info('Using Mixup: {}'.format(mixup))


if len(teacher_dict) > 0:
    logger.info('Using multi teacher')
    teacher_forward_hook_manager_list = []

train_teacher = config["train"].get("train_teacher", False)
ssl_teacher_path = config["train"].get(
    "ssl_teacher_path", "/datad/hungdx/KDW2V-AASISTL/pretrained/xlsr2_300m.pt")
ssl_student_path = config["train"].get(
    "ssl_student_path", "/datad/hungdx/KDW2V-AASISTL/wav2vec_small.pt")

if augment_mode == "rawboost":
    # DEFAULT rawboost 3
    args.algo = config["train"].get("algo", 3)

# Wandb
wandb_disabled = config["train"].get("wandb_disabled", False)
args.batch_size = config['train'].get('batch_size', 32)

if not wandb_disabled:
    wandb.init(project=wandb_project_name, config={
        **config
    }, name=config['name'])

set_random_seed(seed, args)
logger.info('Random seed: {}'.format(seed))


teacher_model = get_model(config['model']['teacher']['name'],
                          device=device, ssl_cpkt_path=ssl_teacher_path).to(device)

if student_model_type == 'ssl':
    if not student_model_name.startswith('Distil_XLSR_N_Trans_Layer') and not student_model_name.startswith('Self_Distil_XLSR_N_Trans_Layer_VIB'):
        student_model = get_model(student_model_name,
                                  device=device, ssl_cpkt_path=ssl_student_path).to(device)
    else:
        student_model = get_model(
            student_model_name, device=device, **config['model']['student']['kwargs']).to(device)
else:
    student_model = get_model(
        student_model_name, d_args=config['model']['student']['kwargs']).to(device)

print("Current number of student parameters: ",  sum(p.numel()
      for p in student_model.parameters()))

teacher_forward_hook_manager = ForwardHookManager(device)
student_forward_hook_manager = ForwardHookManager(device)

student_model = torch.nn.DataParallel(student_model).to(device)

if "is_parallel" in config["model"]["teacher"] and not config["model"]["teacher"]["is_parallel"]:
    logger.info("Teacher model is not parallel")
    teacher_model = teacher_model.to(device)
else:
    logger.info("Teacher model is parallel")
    teacher_model = torch.nn.DataParallel(teacher_model).to(device)

if "pretrained_path" in config["model"]["teacher"]:
    if config["model"]["teacher"]["pretrained_path"] == "":
        logger.info("No pretrained teacher path")
    else:
        try:
            teacher_model.load_state_dict(torch.load(
                config["model"]["teacher"]["pretrained_path"], map_location=device))
            logger.info("Loaded teacher model from {}".format(
                config["model"]["teacher"]["pretrained_path"]))
            
        except:
            logger.info("Failed to load teacher model from {}".format(
                config["model"]["teacher"]["pretrained_path"]))
            sys.exit(0)
else:
    teacher_model.load_state_dict(torch.load(
        args.model_path, map_location=device))
    logger.info("Loaded teacher model from {}".format(args.model_path))
    
# DEBUG
# sys.exit(0)

if "student_resume" in config["train"] and config["train"]["student_resume"] != "":
    student_model.load_state_dict(torch.load(
        config["train"]["student_resume"], map_location=device), strict=False)
    logger.info("Loaded student model from {}".format(
        config["train"]["student_resume"]))


if copy_weights:
    if is_teacher_parallel:
        student_model.module.load_state_dict(
            teacher_model.module.state_dict(), strict=False)
    else:
        student_model.module.load_state_dict(
            teacher_model.state_dict(), strict=False)

    logger.info("Copied teacher weights to student")

    if len(custom_order_copy_weights) > 0 and order == "custom":
        logger.info("Copy transformer weights with custom order")
        for index, value in enumerate(custom_order_copy_weights):
            if is_teacher_parallel:
                student_model.module.ssl_model.model.encoder.layers[index].load_state_dict(
                    teacher_model.module.ssl_model.model.encoder.layers[value].state_dict(), strict=False)
            else:
                student_model.module.ssl_model.model.encoder.layers[index].load_state_dict(
                    teacher_model.ssl_model.model.encoder.layers[value].state_dict(), strict=False)
            logger.info(
                f"Copied teacher transformer weights from  to ssl_model.model.encoder.layers[{value}] to ssl_model.model.encoder.layers[{index}] student")


# Register forward hook
logger.info('Register forward hook for teacher')
for module_path, ios in zip(config['model']['teacher']['teacher_module_paths'], config['model']['teacher']['teacher_module_ios']):
    logger.info('Register teacher forward hook for {}'.format(module_path))
    requires_input, requires_output = ios.split(':')
    requires_input, requires_output = bool(
        requires_input), bool(requires_output)
    if "is_parallel" in config["model"]["teacher"] and not config["model"]["teacher"]["is_parallel"]:
        teacher_forward_hook_manager.add_hook(
            teacher_model, module_path, requires_input=requires_input, requires_output=requires_output)
    else:
        teacher_forward_hook_manager.add_hook(
            teacher_model.module, module_path, requires_input=requires_input, requires_output=requires_output)

# ---------------------------------------- Multi teacher ----------------------------------------
if len(teacher_dict) > 0:
    print(teacher_dict)
    # teacher_dict[teacher_model_name] = teacher_model
    teacher_forward_hook_manager_list.append(teacher_forward_hook_manager)

    for teacher_k, teacher_v in teacher_dict.items():
        if teacher_k != teacher_model_name:  # Skip the teacher model
            key = teacher_v["name"]
            _teacher_forward_hook_manager = ForwardHookManager(device)
            _teacher_model = get_model(
                teacher_v["name"], device=device, ssl_cpkt_path=ssl_teacher_path).to(device)

            if teacher_v["is_parallel"]:
                _teacher_model = torch.nn.DataParallel(
                    _teacher_model).to(device)

            # Load pretrained model

            if "pretrained_path" in teacher_v:
                if teacher_v["pretrained_path"] == "":
                    logger.info("No pretrained teacher path")
                else:
                    try:
                        _teacher_model.load_state_dict(torch.load(
                            teacher_v["pretrained_path"], map_location=device))
                        logger.info("Loaded teacher model from {}".format(
                            teacher_v["pretrained_path"]))
                    except:
                        logger.info("Failed to load teacher model from {}".format(
                            teacher_v["pretrained_path"]))
                        sys.exit(0)
            # Register forward hook
            logger.info('Register forward hook for teacher {}'.format(key))
            for module_path, ios in zip(teacher_v['teacher_module_paths'], teacher_v['teacher_module_ios']):
                logger.info('Register teacher forward hook for {}'.format(
                    module_path))
                requires_input, requires_output = ios.split(':')
                requires_input, requires_output = bool(
                    requires_input), bool(requires_output)
                if teacher_v["is_parallel"]:
                    _teacher_forward_hook_manager.add_hook(
                        _teacher_model.module, module_path, requires_input=requires_input, requires_output=requires_output)
                else:
                    _teacher_forward_hook_manager.add_hook(
                        _teacher_model, module_path, requires_input=requires_input, requires_output=requires_output)

            teacher_module_list.append(_teacher_model)
            teacher_forward_hook_manager_list.append(
                _teacher_forward_hook_manager)

    print("DEBUG")
    sys.exit(0)

logger.info('Register forward hook for student')
for module_path, ios in zip(config['model']['student']['student_module_paths'], config['model']['student']['student_module_ios']):
    logger.info('Register student forward hook for {}'.format(module_path))
    requires_input, requires_output = ios.split(':')
    requires_input, requires_output = bool(
        requires_input), bool(requires_output)
    student_forward_hook_manager.add_hook(
        student_model.module, module_path, requires_input=requires_input, requires_output=requires_output)


# register forward hook for student if recon loss is True
if is_recon_loss:
    logger.info('Register forward hook for student for recon loss')
    student_forward_hook_manager.add_hook(
        student_model.module, 'VIB', requires_input=False, requires_output=True)

logger.info('Prepare training, dev set .....')

logger.info(f'Use {augment_mode} data augmentation')
if dataset:
    logger.info(f'Use {dataset} dataset')


if "sup_contrastive" in config["train"] and config["train"]["sup_contrastive"]:
    logger.info('Use supervised contrastive learning')
    train_loader, dev_loader = get_train_dev_dataloader_contrastive(args)
else:
    train_loader, dev_loader = get_train_dev_dataloader(
        args, augment_mode, dataset)

if not dot:
    optimizer = torch.optim.Adam(student_model.parameters(), lr=float(
        config['train']['learning_rate']), weight_decay=config['train']['weight_decay'])
else:
    '''
        Initialize optimizer for Distillation-Oriented Trainer
    '''
    momentum = float(config['train'].get('momentum', 0.9))
    delta = float(config['train'].get('delta', 0.0075))
    m_task = momentum - delta
    m_kd = momentum + delta
    optimizer = DistillationOrientedTrainer(student_model.parameters(), lr=float(
        config['train']['learning_rate']), momentum=m_task, momentum_kd=m_kd, weight_decay=config['train']['weight_decay'])
    logger.info(
        'Use Distillation-Oriented Trainer with momentum = {} and momentum_kd = {}'.format(m_task, m_kd))

exp_lr_scheduler = None
if 'is_learning_rate_scheduler' in config and config['is_learning_rate_scheduler']:

    # Initialize learning rate scheduler by using its name and its parameters
    exp_lr_scheduler = getattr(torch.optim.lr_scheduler, config['learning_rate_scheduler']['name'])(
        optimizer, **config['learning_rate_scheduler']['params'])
    logger.info(
        f'Use learning rate scheduler {config["learning_rate_scheduler"]["name"]}, {exp_lr_scheduler}')
else:
    logger.info('No learning rate scheduler')


scaler = torch.cuda.amp.GradScaler(enabled=config['train']['amp'])
writer = SummaryWriter('logs/{}'.format(config['name']))
# model_save_path = os.path.join("models", config['name'])

# folder to saved
model_to_save = config['train'].get('model_to_save', 'runs')

if not os.path.exists(model_to_save):
    os.makedirs(model_to_save, exist_ok=True)

model_save_path = os.path.join(model_to_save, config['name'])  # Change to runs

if not os.path.exists(model_save_path):
    os.makedirs(model_save_path)
    logger.info('Created model save path {}'.format(model_save_path))


early_stopping = EarlyStopping(
    patience=config['train']['patience'], verbose=True, model_save_path=model_save_path)

# Train loop
logger.info("Start training")
num_epochs = config['train']['num_epochs']

if train_teacher:
    logger.info('Train teacher model')
    weight = torch.FloatTensor(config['train'].get(
        'cross_entropy_loss_weight', [0.1, 0.9])).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight)
    for epoch in tqdm(range(num_epochs), colour='green'):
        train_loss, train_acc = train_one_epoch(train_loader, teacher_model, criterion,
                                                optimizer, device, scaler, config, exp_lr_scheduler, use_amp=use_amp)
        eval_loss, accuracy = validation(
            dev_loader, teacher_model, criterion, device)

        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/eval', eval_loss, epoch)

        if exp_lr_scheduler is not None and config['learning_rate_scheduler']['name'] == 'StepLR':
            exp_lr_scheduler.step()

        if exp_lr_scheduler is not None and config['learning_rate_scheduler']['name'] != 'ReduceLROnPlateau':
            writer.add_scalar(
                'Lr/epoch', exp_lr_scheduler.get_last_lr()[0], epoch)
            wandb.log({"train_loss": train_loss, "eval_loss": eval_loss, "Eval Accuracy": accuracy,
                       "Train Accuracy": train_acc, "learning_rate": exp_lr_scheduler.get_last_lr()[0]})
        else:
            writer.add_scalar(
                'Lr/epoch', optimizer.param_groups[0]['lr'], epoch)
            wandb.log({"train_loss": train_loss, "eval_loss": eval_loss, "Eval Accuracy": accuracy,
                       "Train Accuracy": train_acc, "learning_rate": optimizer.param_groups[0]['lr']})

        if train_loss < 0.001 and eval_loss < 0.001:
            wandb.alert(
                title='Low loss',
                text=f'train_loss {train_loss} and eval_loss: {eval_loss} is below the acceptable threshold 0.001',
                level=AlertLevel.WARN,
                wait_duration=timedelta(minutes=5)
            )
        # Early stopping
        early_stopping(eval_loss, student_model, epoch)
        if early_stopping.early_stop:
            logger.info("Early stopping")
            break
        # Save model
        if epoch % 1 == 0:
            torch.save(student_model.state_dict(), os.path.join(
                model_save_path, 'checkpoint_{}.pth'.format(epoch)))
            logger.info('Saved model at epoch {}'.format(epoch))
            # Remove old checkpoint
            if epoch > 0:
                old_checkpoint = os.path.join(
                    model_save_path, 'checkpoint_{}.pth'.format(epoch - 1))
                if os.path.exists(old_checkpoint):
                    os.remove(old_checkpoint)
                    logger.info(
                        'Removed old checkpoint {}'.format(old_checkpoint))

    logger.info('End training teacher model')
    sys.exit(0)


if "self_kd_config" in config:
    temperature = float(config['self_kd_config']['temperature'])
    alpha = float(config['self_kd_config']['alpha'])
    beta = float(config['self_kd_config']['beta'])

if 'criterions' in config and 'criterion_weights' in config:
    logger.info('Use mid level loss')
    logger.info('Mid level loss config: {}'.format(config['criterions']))
    logger.info('Mid level loss weight: {}'.format(
        config['criterion_weights']))

start_epoch = 0

if restore:
    logger.info('Restore from previous checkpoint')
    previous_model_saved_path = os.path.join("models", config['name'])

    if not os.path.exists(previous_model_saved_path):
        logger.info(
            'Previous model saved path {} does not exist'.format(previous_model_saved_path))
        sys.exit(0)

    # Get the latest checkpoint startwith 'checkpoint'
    checkpoints = [f for f in os.listdir(
        previous_model_saved_path) if f.startswith('checkpoint')]
    if len(checkpoints) == 0:
        logger.info('No checkpoint found')
        sys.exit(0)
    # Sort the checkpoint by epoch
    checkpoints = sorted(checkpoints, key=lambda x: int(
        x.split('_')[-1].split('.')[0]))
    last_checkpoint = checkpoints[-1]

    student_model_path = os.path.join(
        previous_model_saved_path, last_checkpoint)
    checkpoint = torch.load(student_model_path)
    try:

        student_model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        scaler.load_state_dict(checkpoint['scaler'])
        logger.info(
            'Restore from previous checkpoint at epoch {}'.format(start_epoch))

    except:
        logger.info(
            'Failed to restore from previous checkpoint, the checkpoint may be corrupted or deprecated')
        sys.exit(0)

######## Transformer layer drop ########
if encoder_layerdrop > 0:
    logger.info('Use encoder layer drop')
    student_model.module.ssl_model.model.cfg.encoder_layerdrop = encoder_layerdrop
    logger.info('Encoder layer drop: {}'.format(encoder_layerdrop))


######### Freeze layers #########
if len(freeze_layers) > 0:
    logger.info('Freeze layers')
    for name, param in student_model.module.named_parameters():
        if any(layer in name for layer in freeze_layers):
            param.requires_grad = False
            logger.info(f'Freeze {name}')


# Summary model
summary(student_model, (1, 16000))

for epoch in tqdm(range(start_epoch, num_epochs), colour='green'):
    logger.info('Epoch {}/{}'.format(epoch, num_epochs - 1))

    # Freeze SSL model for the first defined epochs
    if 'freeze_ssl_num_epoch' in config['train'] and epoch < config['train']['freeze_ssl_num_epoch']:
        logger.info('Freeze SSL model')
        student_model.module.ssl_model.frozen()

    else:
        try:
            if student_model.module.ssl_model.freeze:
                logger.info('Unfreeze SSL model')
                student_model.module.ssl_model.unfrozen()
            else:
                # Do nothing
                pass
        except:
            pass

    if "self_kd_config" not in config and byot_kd_train is False:

        train_loss, train_acc, loss_dict = kd_train_epoch(train_loader, student_model, teacher_model, optimizer, device, scaler,
                                                          config, student_forward_hook_manager, teacher_forward_hook_manager, epoch, exp_lr_scheduler=exp_lr_scheduler, use_amp=use_amp)
        eval_loss, accuracy = kd_val_epoch(
            dev_loader, student_model, device, config)
        logger.info(
            'Epoch: {} - train_loss: {} - eval_loss: {}'.format(epoch, train_loss, eval_loss))
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        for key, value in loss_dict.items():
            if isinstance(value, AverageMeter):
                wandb.log({key: value.avg})
            else:
                wandb.log({key: value})
            # writer.add_scalar(f'train_key', value, epoch)

        wandb.log({
            "Accuracy_train": train_acc
        })
    elif byot_kd_train:
        print("BYOT KD training")
        losses, middle1_losses, middle2_losses, middle3_losses, middle4_losses, losses1_kd, losses2_kd, losses3_kd, losses4_kd, feature_losses_1, feature_losses_2, feature_losses_3, feature_losses_4, top1, middle1_top1, middle2_top1, middle3_top1, middle4_top1 = byot_kd_train_epoch(
            train_loader, student_model, optimizer, device, scaler, config,  epoch, exp_lr_scheduler=exp_lr_scheduler, use_amp=use_amp)

        eval_loss, eval_middle1_loss, eval_middle2_loss, eval_middle3_loss, eval_middle4_loss, eval_middle1_kd_loss, eval_middle2_kd_loss, eval_middle3_kd_loss, eval_middle4_kd_loss, eval_feature_loss_1, eval_feature_loss_2, eval_feature_loss_3, eval_feature_loss_4, eval_top1, eval_middle1_top1, eval_middle2_top1, eval_middle3_top1, eval_middle4_top1 = byot_kd_val_epoch(
            dev_loader, student_model, device, config, epoch)

        wandb.log({
            "Train/Accuracy_train": top1.avg,
            "Train/Middle1 Accuracy": middle1_top1.avg,
            "Train/Middle2 Accuracy": middle2_top1.avg,
            "Train/Middle3 Accuracy": middle3_top1.avg,
            "Train/Middle4 Accuracy": middle4_top1.avg,

            "Train/Total Loss": losses.avg,

            "Train/Middle1 Loss": middle1_losses.avg,
            "Train/Middle2 Loss": middle2_losses.avg,
            "Train/Middle3 Loss": middle3_losses.avg,
            "Train/Middle4 Loss": middle4_losses.avg,

            "Train/Middle1 KD Loss": losses1_kd.avg,
            "Train/Middle2 KD Loss": losses2_kd.avg,
            "Train/Middle3 KD Loss": losses3_kd.avg,
            "Train/Middle4 KD Loss": losses4_kd.avg,


            "Train/Feature Loss 1": feature_losses_1.avg,
            "Train/Feature Loss 2": feature_losses_2.avg,
            "Train/Feature Loss 3": feature_losses_3.avg,
            "Train/Feature Loss 4": feature_losses_4.avg,

            "Eval/Accuracy_eval": eval_top1.avg,
            "Eval/Middle1 Accuracy": eval_middle1_top1.avg,
            "Eval/Middle2 Accuracy": eval_middle2_top1.avg,
            "Eval/Middle3 Accuracy": eval_middle3_top1.avg,
            "Eval/Middle4 Accuracy": eval_middle4_top1.avg,

            "Eval/Total Loss": eval_loss.avg,

            "Eval/Middle1 Loss": eval_middle1_loss.avg,
            "Eval/Middle2 Loss": eval_middle2_loss.avg,
            "Eval/Middle3 Loss": eval_middle3_loss.avg,
            "Eval/Middle4 Loss": eval_middle4_loss.avg,

            "Eval/Middle1 KD Loss": eval_middle1_kd_loss.avg,
            "Eval/Middle2 KD Loss": eval_middle2_kd_loss.avg,
            "Eval/Middle3 KD Loss": eval_middle3_kd_loss.avg,
            "Eval/Middle4 KD Loss": eval_middle4_kd_loss.avg,

            "Eval/Feature Loss 1": eval_feature_loss_1.avg,
            "Eval/Feature Loss 2": eval_feature_loss_2.avg,
            "Eval/Feature Loss 3": eval_feature_loss_3.avg,
            "Eval/Feature Loss 4": eval_feature_loss_4.avg,

            "Epoch": epoch
        })

    else:
        train_loss, train_total_label_loss, train_total_kd_loss, train_total_feature_loss, running_total_hidden_rep_loss, running_sup_contrastive_loss = self_KD_teacher_train_epoch(
            train_loader, student_model, teacher_model, optimizer, device, scaler, config, student_forward_hook_manager, teacher_forward_hook_manager, exp_lr_scheduler, temperature=temperature, alpha=alpha, beta=beta, use_amp=use_amp)
        # Eval
        eval_loss, accuracy = self_KD_teacher_val_epoch(
            dev_loader, student_model, device, config)
        writer.add_scalar('Loss/train_label', train_total_label_loss, epoch)
        writer.add_scalar('Loss/train_kd', train_total_kd_loss, epoch)
        writer.add_scalar('Loss/train_feature',
                          train_total_feature_loss, epoch)
        writer.add_scalar('Loss/train_hidden_rep',
                          running_total_hidden_rep_loss, epoch)
        writer.add_scalar('Loss/train_sup_contrastive',
                          running_sup_contrastive_loss, epoch)
        logging.log(logging.INFO, 'Epoch: {} - train_loss: {} - train_total_label_loss: {} - train_total_kd_loss: {} - train_total_feature_loss: {} - running_total_hidden_rep_loss: {} - train_sup_contrastive: {} - eval_loss: {}'.format(
            epoch, train_loss, train_total_label_loss, train_total_kd_loss, train_total_feature_loss, running_total_hidden_rep_loss, running_sup_contrastive_loss, eval_loss))

    if exp_lr_scheduler is not None:
        if config['learning_rate_scheduler']['name'] == 'ReduceLROnPlateau':
            exp_lr_scheduler.step(eval_loss)
        elif config['learning_rate_scheduler']['name'] == 'MultiStepLR' or config['learning_rate_scheduler']['name'] == 'StepLR':
            exp_lr_scheduler.step()

    # Log
    # writer.add_scalar('Loss/train', train_loss, epoch)
    # writer.add_scalar('Loss/eval', eval_loss, epoch)
    # writer.add_scalar('Accuracy/eval', accuracy, epoch)
    # Write current learning rate to tensorboard

    if exp_lr_scheduler is not None and config['learning_rate_scheduler']['name'] != 'ReduceLROnPlateau':
        writer.add_scalar('Lr/epoch', exp_lr_scheduler.get_last_lr()[0], epoch)

        if not byot_kd_train:
            wandb.log({"train_loss": train_loss, "eval_loss": eval_loss, "Eval Accuracy": accuracy,
                       "learning_rate": exp_lr_scheduler.get_last_lr()[0]})
        else:
            wandb.log({"learning_rate": exp_lr_scheduler.get_last_lr()[0]})
    else:
        writer.add_scalar('Lr/epoch', optimizer.param_groups[0]['lr'], epoch)
        if not byot_kd_train:
            wandb.log({"train_loss": train_loss, "eval_loss": eval_loss,
                       "Eval Accuracy": accuracy,
                       "learning_rate": optimizer.param_groups[0]['lr'],
                       "Accuracy_train": train_acc
                       })

    # if train_loss < 0.001 and eval_loss < 0.001:
    #     wandb.alert(
    #         title='Low loss',
    #         text=f'train_loss {train_loss} and eval_loss: {eval_loss} is below the acceptable threshold 0.001',
    #         level=AlertLevel.WARN,
    #         wait_duration=timedelta(minutes=5)
    #     )
    # Early stopping
    if isinstance(eval_loss, AverageMeter):
        early_stopping(eval_loss.avg, student_model, epoch)
    else:
        early_stopping(eval_loss, student_model, epoch)
    if early_stopping.early_stop:
        logger.info("Early stopping")
        break
    # Save model
    if epoch % 1 == 0:
        # Save model and all other necessary information
        save_config = {
            'epoch': epoch,
            'model_state_dict': student_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': eval_loss.avg if isinstance(eval_loss, AverageMeter) else eval_loss,
            # 'accuracy': eval_top1.avg,
            'accuracy': accuracy,
            'scaler': scaler.state_dict(),
        }

        torch.save(save_config, os.path.join(
            model_save_path, 'checkpoint_{}.pth'.format(epoch)))
        logger.info('Saved model at epoch {}'.format(epoch))
        # Remove old checkpoint
        if epoch > 0:
            old_checkpoint = os.path.join(
                model_save_path, 'checkpoint_{}.pth'.format(epoch - 1))
            if os.path.exists(old_checkpoint):
                os.remove(old_checkpoint)
                logger.info('Removed old checkpoint {}'.format(old_checkpoint))


if use_amp:
    logger.info('End automatic mixed precision training')
