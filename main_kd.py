import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import os
import torch
import torch.nn as nn
from trainer import KDTrainer as Trainer
from data.train_set import ASVspoof2019LA as TrainSet
from data.test_set import *
from data.preprocess import PreEmphasis
from logger import Logger
import torch.utils.data as data
import config
import argparse
import yaml
from tqdm import tqdm
import torch.distributed as dist
from utils import find_available_port, set_seed, f_state_dict_wrapper
from models.xlsr_aasist import *
from models.conformer_baseline import Model as ConformerModel
from models.conformer_baseline import MyModel as MyConformerModel
import logging

logger = logging.getLogger()
logging.getLogger("pydub.converter").setLevel(logging.CRITICAL)
logging.getLogger("urllib3.connectionpool").setLevel(logging.CRITICAL)
logging.getLogger("numba.core").setLevel(logging.CRITICAL)
logging.getLogger("git.cmd").setLevel(logging.CRITICAL)
logging.getLogger("fairseq").setLevel(logging.CRITICAL)
logger.disabled = True


def getDataLoader(dataset, batch_size, num_workers):
    return data.DataLoader(dataset=dataset,
                           batch_size=batch_size,
                           pin_memory=True,
                           sampler=DistributedSampler(dataset),
                           num_workers=num_workers
                           )


def run(rank, world_size, port, yml_cfg, args):

    # ------------------------- DDP setup ------------------------- #
    os.environ["MASTER_ADDR"] = 'localhost'
    os.environ["MASTER_PORT"] = port
    init_process_group(backend="nccl", rank=rank, world_size=world_size)
    device = rank

    # ------------------------- import configs ------------------------- #
    is_eval, ckpt = args.is_eval, args.ckpt
    sys_config, exp_config = config.SysConfig(
        yml_cfg['SysConfig']), config.ExpConfig(yml_cfg['ExpConfig'])
    set_seed(exp_config.random_seed)

    # ------------------------- DDP setup ------------------------- #
    logger = Logger(device=device, sys_config=sys_config)

    # ------------------------- Data sets ------------------------- #
    train_set = TrainSet(sys_config=sys_config, exp_config=exp_config)
    train_loader = getDataLoader(
        dataset=train_set, batch_size=exp_config.batch_size_train, num_workers=sys_config.num_workers)

    print("Number of train_loader: ", len(train_loader))

    dev_loader = getDataLoader(dataset=TrainSet(sys_config=sys_config, exp_config=exp_config, is_train=False
                                                ), batch_size=exp_config.batch_size_test, num_workers=sys_config.num_workers)

    print("Number of dev_loader: ", len(dev_loader))
    test_loader = getDataLoader(dataset=ASVspoof2021LA_eval(sys_config=sys_config, exp_config=exp_config, skip_adjustDuration=is_eval
                                                            ), batch_size=exp_config.batch_size_test, num_workers=sys_config.num_workers)

    # ------------------------- set model ------------------------- #
    preprocessor = PreEmphasis(
        device=device, sys_config=sys_config, exp_config=exp_config).to(device)

    if sys_config.model in globals():
        print(sys_config.model)
        model_class = globals()[sys_config.model]
        model = model_class(
            device=device, ssl_cpkt_path='/datad/hungdx/Rawformer-implementation-anti-spoofing/pretrained/xlsr2_300m.pt',
            **exp_config.kwargs
        ).to(device)
    else:
        raise ValueError(f"Model {sys_config.model} not found.")

    # ------------------------- kd_kwargs config ------------------------- #
    kd_kwargs = exp_config.kd_kwargs
    copy_weights = kd_kwargs.get('copy_weights', False)
    custom_order_copy_weights = kd_kwargs["student_kwargs"].get(
        "custom_order", [])
    order = kd_kwargs["student_kwargs"].get(
        "order", None)

    if sys_config.student_model in globals():
        print(sys_config.student_model)
        model_class = globals()[sys_config.student_model]
        student_model = model_class(
            device=device, ssl_cpkt_path='pretrained/xlsr2_300m.pt',
            **kd_kwargs["student_kwargs"]
        ).to(device)
    else:
        raise ValueError(
            f"Student Model {sys_config.student_model} not found.")

    # Print number of parameters
    logger.print(
        f'Number of teacher model parameters: {sum(p.numel() for p in model.parameters())}')

    logger.print(
        f'Number of student model parameters: {sum(p.numel() for p in student_model.parameters())}')

    model = DDP(model, find_unused_parameters=True).to(device)

    # ------------------------- load ckpt ------------------------- #
    if ckpt is not None:
        print(f'Load Teacher checkpoint from {ckpt}')
        state_dict = torch.load(ckpt)

        model.load_state_dict(f_state_dict_wrapper(
            state_dict, data_parallel=True))

    # ------------------------- Define student model ------------------------- #
    student_model = DDP(student_model, find_unused_parameters=True).to(device)

    if copy_weights:
        student_model.module.load_state_dict(
            model.module.state_dict(), strict=False)

        print("Copied teacher weights to student")

        if len(custom_order_copy_weights) > 0 and order == "custom":
            print("Copy transformer weights with custom order")
            for index, value in enumerate(custom_order_copy_weights):

                student_model.module.ssl_model.model.encoder.layers[index].load_state_dict(
                    model.module.ssl_model.model.encoder.layers[value].state_dict(), strict=False)

                print(
                    f"Copied teacher transformer weights from  to ssl_model.model.encoder.layers[{value}] to ssl_model.model.encoder.layers[{index}] student")

    # Deprecated because of a bug in specified parameters where 0 is spoofed and 1 is bonafied
    weight = torch.FloatTensor([0.9, 0.1]).to(device)

    # auto weight
    # because number of bonafide is smaller than number of spoofed samples
    # we need to adjust the weight variable to balance the number of bonafide and spoofed samples
    # For example: if we have 900 spoofed samples and 100 bonafide
    # and in case label is 0 for spoofed and label is 1 for bonafide, the weight should be [0.1, 0.9]
    # total_train_samples = train_set.num_of_spoof + train_set.num_of_bonafide
    # weight = torch.FloatTensor([
    #     train_set.num_of_bonafide / total_train_samples,
    #     train_set.num_of_spoof / total_train_samples
    #     ]).to(device)

    print(f"Current CE weight: {weight}")

    loss_fn = nn.CrossEntropyLoss(weight).to(device)

    # ------------------------- optimizer ------------------------- #
    print(exp_config)
    optimizer = torch.optim.AdamW(params=model.parameters(
    ), lr=exp_config.lr, weight_decay=exp_config.weight_decay)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    #     optimizer,
    #     T_0=exp_config.max_epoch,
    #     T_mult=1,
    #     eta_min=exp_config.lr_min
    # )

    # ------------------------- trainer ------------------------- #
    trainer = Trainer(preprocessor=preprocessor, model=model, student_model=student_model, loss_fn=loss_fn, optimizer=optimizer,
                      train_loader=train_loader, dev_loader=dev_loader, test_loader=test_loader, logger=logger, device=device, exp_config=exp_config, sys_config=sys_config)
    # Initialize with infinity, so any loss is lower
    best_eval_loss = float('inf')
    best_acc = -1.  # Initialize with -1, so any accuracy is higher
    best_loss_epoch = 0
    best_acc_epoch = 0

    if not is_eval:
        for epoch in range(1, exp_config.max_epoch + 1):
            save_checkpoint = False
            logger.print(f'epoch: {epoch}')
            trainer.train()
            # scheduler.step()

            # ------------------------- Validation ------------------------- #
            eval_loss, acc = trainer.test(is_dev=True)
            logger.print(f'Dev acc: {acc}, Dev loss: {eval_loss}')
            logger.wandbLog({'Devacc_LA': acc, 'epoch': epoch})
            logger.wandbLog({'Devloss_LA': eval_loss, 'epoch': epoch})

            # Check for best loss
            if eval_loss < best_eval_loss and acc > 95.0:
                best_eval_loss = eval_loss
                best_loss_epoch = epoch
                save_checkpoint = True  # Flag to save checkpoint

            # Check for best accuracy
            if acc > best_acc:
                best_acc = acc
                best_acc_epoch = epoch
                # Avoid duplication if both best loss and acc are in the same epoch
                if best_acc_epoch != best_loss_epoch and best_acc > 95.0:
                    save_checkpoint = True

            if save_checkpoint and dist.get_rank() == 0:
                # ------------------------- Save model ------------------------- #
                if not os.path.exists(sys_config.path_to_save_model):
                    os.makedirs(sys_config.path_to_save_model, exist_ok=True)

                # Define checkpoint name
                checkpoint_name = f'best_LA_epoch{epoch}_{best_eval_loss:.6f}_{best_acc:.4f}.pt'
                checkpoint_path = os.path.join(
                    sys_config.path_to_save_model, checkpoint_name)
                torch.save(model.state_dict(), checkpoint_path)
                logger.print(f'Model saved to {checkpoint_path}')

                # Reset the flag
                save_checkpoint = False

        # ------------------------- Evaluation ------------------------- #
        # Load the best model
        # trainer.model.load_state_dict(torch.load(
        #     os.path.join(sys_config.path_to_save_model, f'best_LA_epoch{best_epoch}_{best_acc}.pt')))

    else:
        # ------------------------- Evaluation ------------------------- #

        # Load the best model
        logger.print(f'Load the best model from {ckpt}')

        trainer.model.load_state_dict(
            torch.load(ckpt, map_location=f'cuda:{device}'))

    destroy_process_group()


def produce_evaluation_file(dataset, model, device, save_path, batch_size):
    data_loader = data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=4)
    model.eval()
    all_fname_list = []
    all_score_list = []
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    with torch.no_grad():
        for utt_id, batch_x, label in tqdm(data_loader):
            batch_x = batch_x.to(device)
            batch_x = model(batch_x)
            # Get bonafide score
            score = (batch_x[:, 1]).data.cpu().numpy().ravel()
            all_fname_list.extend(utt_id)
            all_score_list.extend(score.tolist())

    # save to file
    with open(save_path, 'w') as fh:
        for f, cm in zip(all_fname_list, all_score_list):
            fh.write('{} {}\n'.format(f, cm))

    print('Done')


if __name__ == "__main__":
    import sys
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/paper.yaml', type=str,
                        help='path to the config file')
    parser.add_argument('--is_eval', action='store_true',
                        help='evaluation mode', default=False)
    parser.add_argument('--eval', default='teacher', type=str,
                        help='Chhose teacher or student to evaluate', required=False)
    parser.add_argument('--ckpt', default=None, type=str,
                        help='path to the checkpoint')
    parser.add_argument('--comment', default=None, type=str,
                        help='Comment to update score name')
    parser.add_argument('--is_score', action='store_true',
                        help='Prodive score file', default=False)
    parser.add_argument('--score_all_folder_path', type=str,
                        help='Prodive score file for all folders', default=None)

    # Auxiliary arguments
    parser.add_argument('--tracks', type=str,
                        default='DF21', help='LA19/LA21/DF21', required=False)

    args = parser.parse_args()

    is_eval, ckpt, is_score, tracks = args.is_eval, args.ckpt, args.is_score, args.tracks

    tracks = tracks.split(',')

    with open(args.config, 'r') as f:
        yml_cfg = yaml.safe_load(f)
    set_seed(config.ExpConfig(yml_cfg['ExpConfig']).random_seed)

    if is_eval:
        yml_cfg['SysConfig']['wandb_disabled'] = True
        eval_mode = args.eval

        if args.score_all_folder_path is not None:
            list_all_checkpoint = os.listdir(args.score_all_folder_path)
            # Ensure that no folder is in the list
            list_all_checkpoint = [
                ckpt for ckpt in list_all_checkpoint if '.pt' in ckpt and not '.txt' in ckpt]
            print(list_all_checkpoint)
            for index, ckpt in enumerate(list_all_checkpoint):
                ckpt = os.path.join(args.score_all_folder_path, ckpt)
                comment = ckpt.split(
                    '_')[-3] + '_' + ckpt.split('_')[-2] + '_' + ckpt.split('_')[-1]

                device = "cuda" if torch.cuda.is_available() else "cpu"
                sys_config, exp_config = config.SysConfig(
                    yml_cfg['SysConfig']), config.ExpConfig(yml_cfg['ExpConfig'])
                logger = Logger(device=device, sys_config=sys_config)
                sys_config.la19_score_save_path = f'{ckpt}_LA19.txt'
                sys_config.la21_score_save_path = f'{ckpt}_LA21.txt'
                sys_config.df21_score_save_path = f'{ckpt}_DF21.txt'

                if sys_config.model in globals() or sys_config.student_model in globals():

                    print(sys_config.model) if eval_mode == 'teacher' else print(
                        sys_config.student_model)
                    kd_kwargs = exp_config.kd_kwargs
                    model_class = globals()[
                        sys_config.model if eval_mode == 'teacher' else sys_config.student_model]

                    result_kwargs = exp_config.kwargs if eval_mode == 'teacher' else kd_kwargs[
                        "student_kwargs"]

                    model = model_class(
                        device=device, ssl_cpkt_path='/datad/hungdx/Rawformer-implementation-anti-spoofing/pretrained/xlsr2_300m.pt',
                        **result_kwargs
                    ).to(device)
                else:
                    raise ValueError(f"Model {sys_config.model} not found.")
                model = torch.nn.DataParallel(model)
                model.load_state_dict(f_state_dict_wrapper(
                    torch.load(ckpt, map_location=device), data_parallel=True))
                print(f'Load model from {ckpt}')
                model = model.module

                # Score for LA19
                print("track: ", tracks)
                for track in tracks:
                    if track == 'LA19':
                        print("Evaluating LA19")
                        if os.path.exists(sys_config.la19_score_save_path):
                            print("File existed, skip")
                            continue
                        test_dataset = ASVspoof2019LA_eval(
                            sys_config=sys_config, exp_config=exp_config)
                        produce_evaluation_file(
                            test_dataset, model, device, sys_config.la19_score_save_path, exp_config.batch_size_test)
                    elif track == 'DF21':
                        # Score for DF21
                        print("Evaluating DF21")
                        if os.path.exists(sys_config.df21_score_save_path):
                            print("File existed, skip")
                            continue
                        test_dataset = ASVspoof2021DF_eval(
                            sys_config=sys_config, exp_config=exp_config)

                        produce_evaluation_file(
                            test_dataset, model, device, sys_config.df21_score_save_path, exp_config.batch_size_test)
                    elif track == 'LA21':
                        print("Evaluating LA21")
                        if os.path.exists(sys_config.la21_score_save_path):
                            print("File existed, skip")
                            continue
                        # Score for LA21
                        test_dataset = ASVspoof2021LA_eval(
                            sys_config=sys_config, exp_config=exp_config)
                        produce_evaluation_file(
                            test_dataset, model, device, sys_config.la21_score_save_path, exp_config.batch_size_test)
                    elif track == 'InTheWild':
                        print("Evaluating InTheWild")

                        # Set attributes for InTheWild
                        setattr(sys_config, 'path_label_in_the_wild',
                                '/datab/Dataset/cnsl_real_fake_audio/in_the_wild.txt')
                        setattr(sys_config, 'path_in_the_wild',
                                '/datab/Dataset/cnsl_real_fake_audio/in_the_wild')
                        setattr(sys_config, 'inthewild_score_save_path',
                                sys_config.df21_score_save_path.replace('DF21', track))
                        
                        if os.path.exists(sys_config.inthewild_score_save_path):
                            print("File existed, skip")
                            continue

                        test_dataset = InTheWild(
                            sys_config=sys_config, exp_config=exp_config)
                        produce_evaluation_file(
                            test_dataset, model, device, sys_config.inthewild_score_save_path, exp_config.batch_size_test)
                    elif track == 'InTheWild1s':
                        print("Evaluating InTheWild1s")

                        # Set attributes for InTheWild
                        setattr(sys_config, 'path_label_in_the_wild',
                                '/data/hungdx/Towards-Real-Time-Deepfake-Speech-Detection-in-Resource-Limited-Scenarios/datasets/prototcols/InTheWild_1s.txt')
                        setattr(sys_config, 'path_in_the_wild',
                                '/data/hungdx/Towards-Real-Time-Deepfake-Speech-Detection-in-Resource-Limited-Scenarios/datasets/InTheWild_1s')
                        setattr(sys_config, 'inthewild_score_save_path',
                                sys_config.df21_score_save_path.replace('DF21', track))
                        
                        if os.path.exists(sys_config.inthewild_score_save_path):
                            print("File existed, skip")
                            continue

                        test_dataset = InTheWild(
                            sys_config=sys_config, exp_config=exp_config)
                        produce_evaluation_file(
                            test_dataset, model, device, sys_config.inthewild_score_save_path, exp_config.batch_size_test)
                    else:
                        raise ValueError('Invalid track')
            sys.exit(0)

        if ckpt is None:
            raise ValueError('ckpt is None')
        if is_score:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            sys_config, exp_config = config.SysConfig(
                yml_cfg['SysConfig']), config.ExpConfig(yml_cfg['ExpConfig'])
            logger = Logger(device=device, sys_config=sys_config)

            if sys_config.model in globals() or sys_config.student_model in globals():

                print(sys_config.model) if eval_mode == 'teacher' else print(
                    sys_config.student_model)
                kd_kwargs = exp_config.kd_kwargs
                model_class = globals()[
                    sys_config.model if eval_mode == 'teacher' else sys_config.student_model]

                result_kwargs = exp_config.kwargs if eval_mode == 'teacher' else kd_kwargs[
                    "student_kwargs"]

                model = model_class(
                    device=device, ssl_cpkt_path='/datad/hungdx/Rawformer-implementation-anti-spoofing/pretrained/xlsr2_300m.pt',
                    **result_kwargs
                ).to(device)
            else:
                raise ValueError(
                    f"Model {sys_config.model} or {sys_config.student_model} not found.")

            model = torch.nn.DataParallel(model)
            model.load_state_dict(f_state_dict_wrapper(
                torch.load(ckpt, map_location=device), data_parallel=True))
            print(f'Load model from {ckpt}')
            model = model.module

            # Update score name if needed
            if args.comment is not None:
                sys_config.la19_score_save_path = sys_config.la19_score_save_path.replace(
                    '.txt', f'_{args.comment}.txt')
                sys_config.la21_score_save_path = sys_config.la21_score_save_path.replace(
                    '.txt', f'_{args.comment}.txt')
                sys_config.df21_score_save_path = sys_config.df21_score_save_path.replace(
                    '.txt', f'_{args.comment}.txt')
            # Score for LA19
            for track in tracks:
                if track == 'LA19':
                    print("Evaluating LA19")
                    if os.path.exists(sys_config.la19_score_save_path):
                        print("File existed, skip")
                        continue
                    test_dataset = ASVspoof2019LA_eval(
                        sys_config=sys_config, exp_config=exp_config)
                    produce_evaluation_file(
                        test_dataset, model, device, sys_config.la19_score_save_path, exp_config.batch_size_test)

                elif track == 'DF21':
                    # Score for DF21
                    print("Evaluating DF21")
                    if os.path.exists(sys_config.df21_score_save_path):
                        print("File existed, skip")
                        continue
                    test_dataset = ASVspoof2021DF_eval(
                        sys_config=sys_config, exp_config=exp_config)

                    produce_evaluation_file(
                        test_dataset, model, device, sys_config.df21_score_save_path, exp_config.batch_size_test)
                elif track == 'LA21':
                    print("Evaluating LA21")
                    if os.path.exists(sys_config.la21_score_save_path):
                        print("File existed, skip")
                        continue
                    # Score for LA21
                    test_dataset = ASVspoof2021LA_eval(
                        sys_config=sys_config, exp_config=exp_config)
                    produce_evaluation_file(
                        test_dataset, model, device, sys_config.la21_score_save_path, exp_config.batch_size_test)
                elif track == 'InTheWild':
                    print("Evaluating InTheWild")

                    # Set attributes for InTheWild
                    setattr(sys_config, 'path_label_in_the_wild',
                            '/datab/Dataset/cnsl_real_fake_audio/in_the_wild.txt')
                    setattr(sys_config, 'path_in_the_wild',
                            '/datab/Dataset/cnsl_real_fake_audio/in_the_wild')
                    setattr(sys_config, 'inthewild_score_save_path',
                            sys_config.df21_score_save_path.replace('DF21', track))

                    test_dataset = InTheWild(
                        sys_config=sys_config, exp_config=exp_config)
                    produce_evaluation_file(
                        test_dataset, model, device, sys_config.inthewild_score_save_path, exp_config.batch_size_test)
                elif track == 'InTheWild1s':
                        print("Evaluating InTheWild1s")

                        # Set attributes for InTheWild
                        setattr(sys_config, 'path_label_in_the_wild',
                                '/data/hungdx/Towards-Real-Time-Deepfake-Speech-Detection-in-Resource-Limited-Scenarios/datasets/prototcols/InTheWild_1s.txt')
                        setattr(sys_config, 'path_in_the_wild',
                                '/data/hungdx/Towards-Real-Time-Deepfake-Speech-Detection-in-Resource-Limited-Scenarios/datasets/InTheWild_1s')
                        setattr(sys_config, 'inthewild_score_save_path',
                                sys_config.df21_score_save_path.replace('DF21', track))
                        
                        if os.path.exists(sys_config.inthewild_score_save_path):
                            print("File existed, skip")
                            continue

                        test_dataset = InTheWild(
                            sys_config=sys_config, exp_config=exp_config)
                        produce_evaluation_file(
                            test_dataset, model, device, sys_config.inthewild_score_save_path, exp_config.batch_size_test)
                else:
                    raise ValueError('Invalid track')
            sys.exit(0)
        # Handle

    print('Running KD training')
    torch.cuda.empty_cache()

    # port = f'10{datetime.datetime.now().microsecond % 100}'
    port = find_available_port(10000, 11000)

    world_size = torch.cuda.device_count()
    mp.spawn(run,
             args=(world_size, port, yml_cfg, args),
             nprocs=world_size
             )
