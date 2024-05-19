import sys
import os
import torch
# torch.set_num_threads(1)
# torch.set_num_interop_threads(1)
from torch import nn
from torch.utils.data import DataLoader
from data_utils import *
from tensorboardX import SummaryWriter
from startup_config import set_random_seed
from student import *
from teacher import *
from kdtoolkit import *
from menu import get_main_menu
from utils import EarlyStopping
from torch.optim.lr_scheduler import StepLR
from torchaudio.models.wav2vec2.utils import import_fairseq_model
from tqdm import tqdm

import logging

# Get the Numba logge;p0./r
logger = logging.getLogger('numba')
logger.setLevel(logging.WARNING)  # Set level to WARNING, ERROR, or CRITICAL

__author__ = "Hungdx"
__email__ = "hungdx@soongsil.ac.kr"

# use train


def evaluate_accuracy(dev_loader, model, device, kd_method=None):
    val_loss = 0.0
    num_total = 0.0
    model.eval()
    weight = torch.FloatTensor([0.1, 0.9]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight)
    with torch.no_grad():
        for batch_x, batch_y in dev_loader:

            batch_size = batch_x.size(0)
            num_total += batch_size
            batch_x = batch_x.to(device)
            batch_y = batch_y.view(-1).type(torch.int64).to(device)

            if kd_method == 'KD_logits':
                batch_out = model(batch_x)
            elif kd_method == 'KD_cosine':
                batch_out, _ = model(batch_x)
            elif kd_method == 'KD_mse':
                batch_out, _ = model(batch_x)
            else:
                batch_out = model(batch_x)

            batch_loss = criterion(batch_out, batch_y)
            val_loss += (batch_loss.item() * batch_size)

        val_loss /= num_total

    return val_loss


def produce_evaluation_file(dataset, model, device, save_path, kd_method=None, batch_size=4, is_half=False):
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False,
                             pin_memory=True if device != "cpu" else False, pin_memory_device=device)
    model.eval()
    fname_list = []
    score_list = []

    with torch.no_grad():
        for batch_x, utt_id in tqdm(data_loader):
            fname_list = []
            score_list = []
            batch_size = batch_x.size(0)

            if is_half:
                batch_x = batch_x.half()
            batch_x = batch_x.to(device)

            if kd_method == 'KD_logits':
                batch_out = model(batch_x)
            elif kd_method == 'KD_cosine':
                batch_out, _ = model(batch_x)
            elif kd_method == 'KD_mse':
                batch_out, _ = model(batch_x)
            elif kd_method == 'self_KD':
                batch_out, spectral_output, temporal_output, graph_output_S, graph_output_T, hs_gal_output_S, hs_gal_output_T, middle_feature1, middle_feature2, final_feature1, final_feature2 = model(
                    batch_x)
            elif kd_method == 'self_KD_Teacher':
                batch_out, spectral_output, temporal_output, graph_output_S, graph_output_T, hs_gal_output_S, hs_gal_output_T, middle_feature1, middle_feature2, final_feature1, final_feature2, hidden_features = model(
                    batch_x)
            else:
                batch_out = model(batch_x)

            if device == 'cpu':
                # batch_out = batch_out.to(torch.float32)
                # Move to CPU and detach from the computation graph
                batch_score = batch_out[:, 1].cpu().detach()

            else:
                batch_score = (batch_out[:, 1]
                               ).data.cpu().numpy(force=True).ravel()
            # add outputs
            fname_list.extend(utt_id)
            score_list.extend(batch_score.tolist())

            with open(save_path, 'a+') as fh:

                for f, cm in zip(fname_list, score_list):
                    fh.write('{} {}\n'.format(f, cm))
            fh.close()
    print('Scores saved to {}'.format(save_path))
# use train


def train_epoch(train_loader, model, optimizer, device):
    running_loss = 0

    num_total = 0.0

    model.train()

    # set objective (Loss) functions
    weight = torch.FloatTensor([0.1, 0.9]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight)

    for batch_x, batch_y in train_loader:

        batch_size = batch_x.size(0)
        num_total += batch_size

        batch_x = batch_x.to(device)

        # 1, 0
        batch_y = batch_y.view(-1).type(torch.int64).to(device)
        batch_out = model(batch_x)

        batch_loss = criterion(batch_out, batch_y)

        running_loss += (batch_loss.item() * batch_size)

        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()

    running_loss /= num_total

    return running_loss


class W2V2_TA(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def extract_feat(self, x):
        feat, _ = self.model(x)
        return feat

    def forward(self, x):
        return self.extract_feat(x)


if __name__ == '__main__':

    if not os.path.exists('models'):
        os.mkdir('models')
    args = get_main_menu()

    # make experiment reproducible
    set_random_seed(args.seed, args)

    track = args.track

    assert track in ['LA', 'PA', 'DF'], 'Invalid track given'

    # database
    prefix_2021 = 'ASVspoof2021.{}'.format(track)

    # define model saving path
    model_tag = 'model_{}_{}_{}_{}_{}'.format(
        track, args.loss, args.num_epochs, args.batch_size, args.lr)
    if args.comment:
        model_tag = model_tag + '_{}'.format(args.comment)
    model_save_path = os.path.join('models', model_tag)

    # set model save directory
    if not os.path.exists(model_save_path):
        os.mkdir(model_save_path)

    # GPU device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Device: {}'.format(device))

    if args.KD_logits:
        model = W2V2_AASIST(device)
        if args.ssl_type == 'Distil_XLSR':
            student = Distil_W2V2_AASISTL(device)
        elif args.ssl_type == 'ft':
            # W2V2BASE 95M fine-tuned
            student = Distil_W2V2FTBASE_AASISTL(device)
        else:
            # W2V2BASE 95M
            student = Distil_W2V2BASE_AASISTL(device)
        kd_method = 'KD_logits'

    elif args.KD_cosine:
        model = W2V2_AASIST_Cosine(device)
        if args.ssl_type == 'Distil_XLSR':
            student = Distil_W2V2_AASISTL_Cosine(device)
        else:
            # W2V2BASE 95M
            student = Distil_W2V2BASE_AASISTL_Cosine(device)
        kd_method = 'KD_cosine'

    elif args.KD_mse:
        model = W2V2_AASIST_Regressor()
        if args.ssl_type == 'Distil_XLSR':
            student = Distil_W2V2_AASISTL_Regressor(device)
        else:
            # W2V2BASE 95M
            student = Distil_W2V2BASE_AASISTL_Regressor(device)
        kd_method = 'KD_mse'
    elif args.self_KD:
        if args.self_KD_type == 'self_KD':
            student = Distil_W2V2BASE_AASISTL_Self_KD(device)
            kd_method = 'self_KD'
        elif args.self_KD_type == 'self_KD_Teacher':
            student = Distil_W2V2BASE_AASISTL_Self_KD_Teacher(device)
            kd_method = 'self_KD_Teacher'
        elif args.self_KD_type == 'self_KD_HG':
            student = Distil_W2V2BASEHG_AASISTL_Self_KD(device)
            kd_method = 'self_KD_Teacher'
        elif args.self_KD_type == 'self_KD_Teacher_HG':
            model = W2V2_AASIST_Cosine(device)
            student = Distil_W2V2BASEHG_AASISTL_Self_KD_Teacher(device)
            kd_method = 'self_KD_Teacher'
        elif args.self_KD_type == 'self_KD_Teacher_HG_Cosine_Dropout':
            model = W2V2_AASIST_Cosine(device)
            student = Distil_W2V2BASEHG_AASISTL_Self_KD_Teacher_Drop(device)
            kd_method = 'self_KD_Teacher_Dropout'
        elif args.self_KD_type == 'self_KD_Teacher_SSL_WAV2VEC2_BASE_TA':
            model = W2V2_AASIST_Cosine(device)
            student = Distil_SSL_WAV2VEC2_TA_Self_KD_Teacher(device, fe='base')
            kd_method = 'self_KD_Teacher'
            print('self_KD_Teacher_SSL_WAV2VEC2_BASE_TA')
        elif args.self_KD_type == 'self_KD_Teacher_SSL_WAV2VEC2_ASR_BASE_960H_TA':
            model = W2V2_AASIST_Cosine(device)
            student = Distil_SSL_WAV2VEC2_TA_Self_KD_Teacher(
                device, fe='SSL_WAV2VEC2_ASR_BASE_960H_TA')
            kd_method = 'self_KD_Teacher'
            print('self_KD_Teacher_SSL_WAV2VEC2_ASR_BASE_960H_TA')
        elif args.self_KD_type == 'self_KD_Teacher_SSL_WAV2VEC2_ASR_BASE_960H_TA_v2':
            model = W2V2_AASIST_Cosine(device)
            student = Distil_SSL_WAV2VEC2_TA_Self_KD_Teacher(
                device, fe='SSL_WAV2VEC2_ASR_BASE_960H_TA')
            kd_method = 'self_KD_Teacher'
            print('self_KD_Teacher_SSL_WAV2VEC2_ASR_BASE_960H_TA_v2')
        elif args.self_KD_type == 'self_KD_Teacher_SSL_WAV2VEC2_BASE_FSTA':
            model = W2V2_AASIST_Cosine(device)
            student = Distil_SSL_WAV2VEC2_TA_Self_KD_Teacher(
                device, fe='SSL_WAV2VEC2_BASE_FSTA')
            kd_method = 'self_KD_Teacher'
        elif args.self_KD_type == 'self_KD_Teacher_Distil_SSL_WAV2VEC2_BASE_TAHG':
            model = W2V2_AASIST_Cosine(device)
            student = Distil_SSL_WAV2VEC2_TA_Self_KD_Teacher(
                device, fe='Distil_SSL_WAV2VEC2_BASE_TAHG')
            kd_method = 'self_KD_Teacher'
        elif args.self_KD_type == 'self_KD_SSL_WAV2VEC2_BASE_HF':
            model = W2V2_AASIST_Cosine(device)
            student = Distil_SSL_WAV2VEC2_TA_Self_KD_Teacher(
                device, fe='SSL_WAV2VEC2_BASE_HF')
            kd_method = 'self_KD_Teacher'
        elif args.self_KD_type == 'self_KD_Teacher_SSL_WAV2VEC2_BASE_960H_HF':
            model = W2V2_AASIST_Cosine(device)
            student = Distil_SSL_WAV2VEC2_TA_Self_KD_Teacher(
                device, fe='SSL_WAV2VEC2_BASE_960H_HF')
            kd_method = 'self_KD_Teacher'
        # if not args.self_KD_type == 'self_KD_Teacher_HG':
        #     kd_method = 'self_KD'
        # elif args.self_KD_type == 'self_KD_Teacher_HG_Cosine_Dropout':
        #     kd_method = 'self_KD_Teacher_Dropout'
        # else:
        #     kd_method = 'self_KD_Teacher'
    else:
        raise ValueError('Invalid KD method given')

    # print model parameters
    if not args.self_KD or args.self_KD_type in ['self_KD_Teacher_HG', 'self_KD_Teacher_HG_Cosine_Dropout', 'self_KD_Teacher_SSL_WAV2VEC2_BASE_TA', 'self_KD_Teacher_SSL_WAV2VEC2_ASR_BASE_960H_TA', 'self_KD_Teacher_SSL_WAV2VEC2_ASR_BASE_960H_TA_v2', 'self_KD_Teacher_Distil_SSL_WAV2VEC2_BASE_TAHG', 'self_KD_Teacher_SSL_WAV2VEC2_BASE_FSTA', 'self_KD_SSL_WAV2VEC2_BASE_HF', 'self_KD_Teacher_SSL_WAV2VEC2_BASE_960H_HF']:
        nb_params = sum([param.view(-1).size()[0]
                        for param in model.parameters()])
        model = nn.DataParallel(model).to(device)
        print('Teacher nb_params:', nb_params)

    nb_params = sum([param.view(-1).size()[0]
                    for param in student.parameters()])
    student = nn.DataParallel(student).to(device)
    print('Student nb_params:', nb_params)

    # set Adam optimizer
    optimizer = torch.optim.Adam(
        student.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=args.use_amp)
    # set learning rate scheduler
    lr_scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

    if args.model_path:
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        print('AASIST Model loaded : {}'.format(args.model_path))

    if args.student_ckpt:
        student.load_state_dict(torch.load(
            args.student_ckpt, map_location=device))
        print('Student model loaded : {}'.format(args.student_ckpt))

    if args.student_restore:
        try:
            # Restore student model from best checkpoint
            cpt = sorted(os.listdir(model_save_path), key=lambda x: int(
                x.split('_')[2].split('.')[0]) if not x.startswith('epoch') else 0)[-1]
            print('Loading student model from {}'.format(
                os.path.join(model_save_path, cpt)))
            # Restore student model from last checkpoint
            # last_cpt = sorted(os.listdir(model_save_path), key=lambda x: int(x.split('_')[1].split('.')[0]) if not x.startswith('best') else 0 )[-1]

            student.load_state_dict(torch.load(
                os.path.join(model_save_path, cpt)))
            print('Student model loaded : {}'.format(
                os.path.join(model_save_path, cpt)))
            # print('Training from epoch {}'.format(int(last_cpt.split('_')[1].split('.')[0])))
        except Exception as e:
            print('No checkpoint student found in ', model_save_path)
            print(e)
            print('Training from scratch')

    if args.student_model_path:
        print('Loading student model from {}'.format(args.student_model_path))
        try:
            last_cpkt = torch.load(
                args.student_model_path, map_location=device)
            student.load_state_dict(last_cpkt["model"])
            optimizer.load_state_dict(last_cpkt["optimizer"])
            scaler.load_state_dict(last_cpkt["scaler"])
            print('Student model loaded : {}'.format(args.student_model_path))
        except Exception as e:
            print('No checkpoint student found in ', args.student_model_path)
            print(e)
            print('Training from scratch')

    if args.half:
        student = student.half().to(device)
        print('Student Model casted to half precision to evaluate')
    # evaluation
    if args.eval:
        _, file_eval = genSpoof_list(dir_meta=os.path.join(args.protocols_path+'ASVspoof_{}_cm_protocols/{}.cm.eval.trl.txt'.format(
            track, prefix_2021)), is_train=False, is_eval=True, num_eval_samples=args.num_eval_samples)
        print('no. of eval trials', len(file_eval))
        eval_set = Dataset_ASVspoof2021_eval(list_IDs=file_eval, base_dir=os.path.join(
            args.database_path+'ASVspoof2021_{}_eval/'.format(args.track)))

        # Produce evaluation file
        if args.is_eval_teacher:
            model.module.ssl_model = W2V2_TA(import_fairseq_model(
                model.module.ssl_model.model)).to(device)
        # Choose model to evaluate
        # if args.is_eval_teacher:
        #     print('Evaluating teacher model')
        #     model_to_eval = model
        # else:
        #     print('Evaluating student model')
        #     model_to_eval = student

        # Cast model to half precision if needed
        print("Current KD method: {}".format(kd_method))

        produce_evaluation_file(eval_set, model if args.is_eval_teacher else student, device,
                                args.eval_output, batch_size=args.batch_size_eval, kd_method=kd_method, is_half=args.half)
        sys.exit(0)

    # define train dataloader
    d_label_trn, file_train = genSpoof_list(dir_meta=os.path.join(
        args.protocols_path+'ASVspoof_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt'), is_train=True, is_eval=False)
    train_loader, dev_loader = get_train_dev_dataloader(args)
    # Training and validation
    start_epoch = 0 if not args.student_restore else int(
        cpt.split('_')[2].split('.')[0]) + 1
    assert start_epoch == 0 or type(
        start_epoch) == int, 'Invalid start epoch given'
    if args.student_model_path:
        try:
            # Get last element of student_model_path in uri format

            path_to_split = args.student_model_path.split('/')[-1]

            start_epoch = int(path_to_split.split('_')[-1].split('.')[0]) + 1
        except Exception as e:
            print('No checkpoint student found in ', args.student_model_path)
            print(e)
            print('Training from scratch')
    print('Start epoch: {}'.format(start_epoch))

    num_epochs = args.num_epochs
    writer = SummaryWriter('logs/{}'.format(model_tag))
    early_stopping = EarlyStopping(
        patience=args.patience, verbose=True, model_save_path=model_save_path)
    if args.use_amp:
        print('Using automatic mixed precision training')

    for epoch in range(start_epoch, num_epochs):
        if args.KD_logits:
            if args.ssl_type != 'Distil_XLSR' and args.ssl_type != 'ft':
                running_loss = train_knowledge_distillation(
                    model, student, train_loader, optimizer, T=4.10431, soft_target_loss_weight=0.498657, ce_loss_weight=0.640261, device=device)
            else:
                running_loss = train_knowledge_distillation(
                    model, student, train_loader, optimizer, T=2, soft_target_loss_weight=0.25, ce_loss_weight=0.75, device=device)
            KD_method = 'KD_logits'
        elif args.KD_cosine:
            if args.ssl_type != 'Distil_XLSR':
                running_loss = train_kd_cosine_loss(
                    model, student, train_loader, optimizer, hidden_rep_loss_weight=0.933, ce_loss_weight=0.059, device=device)
            else:
                running_loss = train_kd_cosine_loss(
                    model, student, train_loader, optimizer, hidden_rep_loss_weight=0.25, ce_loss_weight=0.75, device=device)
            KD_method = 'KD_cosine'
        elif args.KD_mse:
            if args.ssl_type != 'Distil_XLSR':
                running_loss = train_kd_mse_loss(
                    model, student, train_loader, optimizer, feature_map_weight=0.742208, ce_loss_weight=0.483691, device=device)
            else:
                running_loss = train_kd_mse_loss(
                    model, student, train_loader, optimizer, feature_map_weight=0.25, ce_loss_weight=0.75, device=device)
            KD_method = 'KD_mse'
        elif args.self_KD:
            SELF_KD_TEACHER_SUPPORT_LIST = ['self_KD_Teacher_HG', 'self_KD_Teacher_SSL_WAV2VEC2_BASE_TA', 'self_KD_Teacher_SSL_WAV2VEC2_ASR_BASE_960H_TA',
                                            'self_KD_Teacher_Distil_SSL_WAV2VEC2_BASE_TAHG', 'self_KD_Teacher_SSL_WAV2VEC2_BASE_FSTA', 'self_KD_SSL_WAV2VEC2_BASE_HF', 'self_KD_Teacher_SSL_WAV2VEC2_BASE_960H_HF']
            if args.self_KD_type in SELF_KD_TEACHER_SUPPORT_LIST:
                running_loss, running_total_label_loss, running_total_kd_loss, running_total_feature_loss = self_KD_teacher_train_epoch(
                    train_loader, student, model, optimizer, device, scaler, use_amp=args.use_amp)
                val_loss = self_KD_teacher_val_epoch(
                    dev_loader, student, device)
                writer.add_scalar('running_total_label_loss',
                                  running_total_label_loss, epoch)
                writer.add_scalar('running_total_kd_loss',
                                  running_total_kd_loss, epoch)
                writer.add_scalar('running_total_feature_loss',
                                  running_total_feature_loss, epoch)

            elif args.self_KD_type == 'self_KD_Teacher_HG_Cosine_Dropout':
                running_loss, running_total_label_loss, running_total_kd_loss, running_total_feature_loss, running_total_kl_loss = self_KD_Dropout_train_epoch(
                    train_loader, model, student, optimizer, device, scaler, lr_scheduler, use_amp=args.use_amp)
                val_loss = self_KD_teacher_val_epoch(
                    dev_loader, student, device, kd_method='self_KD_Teacher_Dropout')
                writer.add_scalar('running_total_label_loss',
                                  running_total_label_loss, epoch)
                writer.add_scalar('running_total_kd_loss',
                                  running_total_kd_loss, epoch)
                writer.add_scalar('running_total_feature_loss',
                                  running_total_feature_loss, epoch)
                writer.add_scalar('running_total_kl_loss',
                                  running_total_kl_loss, epoch)

            elif args.self_KD_type == 'self_KD_Teacher_SSL_WAV2VEC2_ASR_BASE_960H_TA_v2':
                running_loss, running_total_label_loss, running_total_kd_loss, running_total_feature_loss = self_KD_teacher2_train_epoch(
                    train_loader, student, model, optimizer, device, scaler, use_amp=args.use_amp)
                val_loss = self_KD_teacher_val_epoch(
                    dev_loader, student, device)
                writer.add_scalar('running_total_label_loss',
                                  running_total_label_loss, epoch)
                writer.add_scalar('running_total_kd_loss',
                                  running_total_kd_loss, epoch)
                writer.add_scalar('running_total_feature_loss',
                                  running_total_feature_loss, epoch)

            else:
                running_loss, running_total_label_loss, running_total_kd_loss, running_total_feature_loss = self_KD_teacher_train_epoch(
                    train_loader, student, model, optimizer, device, scaler, use_amp=args.use_amp)
                val_loss, eval_accuracy = self_KD_val_epoch(
                    dev_loader, student, device)
                writer.add_scalar('running_total_label_loss',
                                  running_total_label_loss, epoch)
                writer.add_scalar('running_total_kd_loss',
                                  running_total_kd_loss, epoch)
                writer.add_scalar('running_total_feature_loss',
                                  running_total_feature_loss, epoch)

            KD_method = 'self_KD'
        else:
            logging.log(logging.ERROR, 'Invalid KD method given')
            raise ValueError('Invalid KD method given')

        # Validate student and save model
        if not args.self_KD:
            val_loss = evaluate_accuracy(
                dev_loader, student, device, kd_method=KD_method)

        writer.add_scalar('loss', running_loss, epoch)
        writer.add_scalar('val_loss', val_loss, epoch)

        print('\n{} - {} - {} '.format(epoch, running_loss, val_loss))

        if args.use_amp:
            checkpoint = {"model": student.state_dict(),
                          "optimizer": optimizer.state_dict(),
                          "scaler": scaler.state_dict()}
            torch.save(checkpoint, os.path.join(
                model_save_path, 'epoch_{}.pth'.format(epoch)))

        else:
            torch.save(student.state_dict(), os.path.join(
                model_save_path, 'epoch_{}.pth'.format(epoch)))

        # Remove previous checkpoint
        if epoch > 0:
            prev_checkpoint = os.path.join(
                model_save_path, 'epoch_{}.pth'.format(epoch - 1))
            if os.path.exists(prev_checkpoint):
                os.remove(prev_checkpoint)

        # early_stopping needs the validation loss to check if it has decresed,
        # and if it has, it will make a checkpoint of the current model
        early_stopping(val_loss, student, epoch)

        if early_stopping.early_stop:
            logging.log(logging.INFO, "Early stopping")
            break

    print('Finished Training')
