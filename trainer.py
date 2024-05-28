import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn import metrics
from scipy.optimize import brentq
from scipy.interpolate import interp1d
import torch.distributed as dist
from ddp_util import all_gather
import logger
import config
from data.augmentation import WaveformAugmetation
import numpy as np
import os
import torch.distributed as dist
# from torchdistill.core.forward_hook import ForwardHookManager
# from torchdistill.losses.registry import get_mid_level_loss
from utils import AverageMeter


class Trainer:

    def __init__(self, preprocessor: nn.Module, model: nn.Module, loss_fn: nn.Module, optimizer, train_loader, dev_loader, test_loader, logger: logger.Logger, device, exp_config, sys_config):

        self.preprocessor = preprocessor
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        # self.score_save_path = sys_config.score_save_path
        self.train_loader = train_loader
        self.dev_loader = dev_loader
        self.test_loader = test_loader
        self.exp_config = exp_config
        self.logger = logger

        self.device = device

        # --------------- settings for DA --------------- #
        self.allow_data_augmentation = exp_config.allow_data_augmentation  # True of False
        self.waveform_augmentation = WaveformAugmetation(
            exp_config.data_augmentation)

    def train(self):

        self.model.train()
        self.loss_fn.train()
        num_correct = 0.0
        iter_count = 0
        num_total = 0.0
        loss_sum = 0
        num_item_train = len(self.train_loader)
        pbar = tqdm(self.train_loader)
        for _, x, label in pbar:

            self.optimizer.zero_grad(set_to_none=True)

            x, label = x.to(self.device), label.to(
                device=self.device, dtype=torch.int64)
            batch_size = x.size(0)
            num_total += batch_size
            x = self.preprocessor(x)

            # augmentation
            if self.allow_data_augmentation:
                x = self.waveform_augmentation(x)

            # forward and backward
            x = self.model(x)
            loss = self.loss_fn(x, label)

            loss.backward()
            self.optimizer.step()

            # logging
            loss = loss.detach()
            iter_count += 1
            loss_sum += loss

            pbar.set_description(f'loss: {loss}')
            _, batch_pred = x.max(dim=1)
            num_correct += (batch_pred == label).sum(dim=0).item()

            if num_item_train * 0.02 <= iter_count:
                self.logger.wandbLog({'Loss': loss_sum / float(iter_count)})
                loss_sum = 0
                iter_count = 0
        self.logger.wandbLog(
            {'Train Acc': (num_correct / num_total) * 100})

    def test(self, is_dev=False):

        return self._test(self.dev_loader, mode='validation') if is_dev else self._test(self.test_loader)

    def _test(self, test_loader, mode='evaluation'):
        self.model.eval()

        scores = []
        labels = []
        num_correct = 0.0
        iter_count = 0
        num_total = 0.0

        pbar = tqdm(test_loader, mode)
        loss_sum = 0
        iter_count = 0
        with torch.no_grad():
            for utter_id, x, label in pbar:
                iter_count += 1
                x, label = x.to(
                    self.device), label.view(-1).type(torch.int64).to(self.device)
                batch_size = x.size(0)
                num_total += batch_size
                x = self.preprocessor(x)

                x = self.model(x)

                loss = self.loss_fn(x, label)
                _, batch_pred = x.max(dim=1)
                num_correct += (batch_pred == label).sum(dim=0).item()

                loss_sum += (loss.item() * batch_size)
                # scores.append(x.cpu())
                # labels.append(label.cpu())

            # scores = torch.cat(scores, dim=0)
            # labels = torch.cat(labels, dim=0)

        # scores = all_gather(scores)
        # labels = all_gather(labels)
        eval_loss = loss_sum / num_total
        # self.logger.wandbLog({'Eval Loss': loss_sum / float(iter_count)})
        # # Convert scores and labels list to numpy
        # scores = np.array(scores)
        # labels = np.array(labels)

        # eer = self.calculate_EER(scores, labels)
        # print(f'EER: {eer}')
        accuracy = (num_correct / num_total) * 100
        self.logger.wandbLog(
            {'Dev Acc': accuracy, 'Dev Loss': eval_loss})
        return eval_loss, accuracy

    def calculate_EER(self, scores, labels):
        # Save the scores and labels

        fpr, tpr, _ = metrics.roc_curve(labels, scores, pos_label=1)
        eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
        return eer * 100


class KDTrainer:

    def __init__(self, preprocessor: nn.Module, model: nn.Module, student_model: nn.Module,
                 loss_fn: nn.Module, optimizer, train_loader, dev_loader, test_loader,
                 logger: logger.Logger, device, exp_config, sys_config):

        self.preprocessor = preprocessor
        self.model = model
        self.student_model = student_model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        # self.score_save_path = sys_config.score_save_path
        self.train_loader = train_loader
        self.dev_loader = dev_loader
        self.test_loader = test_loader
        self.teacher_forward_hook_manager = ForwardHookManager(
            f'cuda:{device}')
        self.student_forward_hook_manager = ForwardHookManager(
            f'cuda:{device}')
        self.exp_config = exp_config

        # Register forward hooks for teacher and student
        self.register_forward_hook(
            self.exp_config, 'teacher')
        self.register_forward_hook(
            self.exp_config,  'student')

        self.logger = logger
        self.device = device

        # --------------- settings for DA --------------- #
        self.allow_data_augmentation = exp_config.allow_data_augmentation  # True of False
        self.waveform_augmentation = WaveformAugmetation(
            exp_config.data_augmentation)

    def register_forward_hook(self, config,  model_type):
        # print(config.kd_kwargs['model'][model_type])
        for module_path, ios in zip(config.kd_kwargs['model'][model_type][f'{model_type}_module_path'], config.kd_kwargs['model'][model_type][f'{model_type}_module_ios']):
            print(f'Register {model_type} forward hook for {module_path}')
            requires_input, requires_output = ios.split(':')
            try:
                requires_input, requires_output = requires_input.lower(
                ) == 'true', requires_output.lower() == 'true'
            except ValueError:
                print(
                    f"Invalid value for requires_input or requires_output for {model_type} at {module_path}")
                continue
            print(
                f'Add hook to {model_type} model at {module_path} with requires_input={requires_input} and requires_output={requires_output}')
            if model_type == 'teacher':
                self.teacher_forward_hook_manager.add_hook(
                    self.model, module_path, requires_input=requires_input, requires_output=requires_output)
            else:
                self.student_forward_hook_manager.add_hook(
                    self.student_model, module_path, requires_input=requires_input, requires_output=requires_output)

    def train(self):

        self.model.eval()
        self.student_model.train()
        self.loss_fn.train()

        num_correct = 0.0
        iter_count = 0
        num_total = 0.0
        loss_sum = 0
        num_item_train = len(self.train_loader)
        pbar = tqdm(self.train_loader)

        loss_dict = dict()
        loss_dict['ce_loss'] = AverageMeter()
        loss_dict['total_loss'] = AverageMeter()
        ce_loss_weight = self.exp_config.kd_kwargs.get('ce_loss_weight', 1.0)

        criterions = self.exp_config.kd_kwargs.get('kd_criterions', [])
        # use_amp = self.exp_config.kd_kwargs.get('use_amp', False)
        criterion_key_list = []
        for loss in criterions:
            # loss_dict[f"{loss['key']}_{loss['kwargs']['student_module_path']}_{loss['kwargs']['teacher_module_path']}"] = 0

            student_module_path = loss.get('kwargs', {}).get(
                'student_module_path', 'default_student_module_path')
            teacher_module_path = loss.get('kwargs', {}).get(
                'teacher_module_path', 'default_teacher_module_path')
            key = loss.get('key', 'default_key')
            criterion_key = f"{key}_{student_module_path}_{teacher_module_path}"
            criterion_key_list.append(criterion_key)
            loss_dict[criterion_key] = AverageMeter()

        for _, x, label in pbar:

            self.optimizer.zero_grad(set_to_none=True)

            x, label = x.to(self.device), label.to(
                device=self.device, dtype=torch.int64)
            batch_size = x.size(0)
            num_total += batch_size
            x = self.preprocessor(x)

            # Define losses
            # ce_loss = torch.tensor(0.).to(self.device)
            kd_loss = torch.tensor(0.).to(self.device)

            # augmentation
            if self.allow_data_augmentation:
                x = self.waveform_augmentation(x)

            # Get teacher outputs
            with torch.no_grad():
                # print(f'Input shape: {x.shape}')
                _ = self.model(x)
                teacher_io_dict = self.teacher_forward_hook_manager.pop_io_dict()

            # forward
            x = self.student_model(x)
            # pop student io dict after forward pass
            student_io_dict = self.student_forward_hook_manager.pop_io_dict()
            ce_loss = self.loss_fn(x, label)  # CE loss

            ####### KD Losses #######
            for loss, weight, criterion_key in zip(self.exp_config.kd_kwargs['kd_criterions'], self.exp_config.kd_kwargs['kd_criterion_weights'], criterion_key_list):
                weight = float(weight)
                loss_i = get_mid_level_loss(
                    mid_level_criterion_config=loss)

                tmp_loss = (loss_i.forward(student_io_dict,
                                           teacher_io_dict, label) * weight)
                tmp_loss_weight = tmp_loss * weight
                loss_dict[criterion_key].update(
                    tmp_loss_weight.item(), batch_size)
                kd_loss += tmp_loss_weight

            loss_dict['ce_loss'].update(ce_loss.item(), batch_size)

            total_loss = ce_loss_weight * ce_loss + kd_loss

            loss_dict['total_loss'].update(total_loss.item(), batch_size)

            total_loss.backward()
            self.optimizer.step()

            # logging
            total_loss = total_loss.detach()
            iter_count += 1

            loss_sum += total_loss

            pbar.set_description(f'loss: {total_loss}')
            _, batch_pred = x.max(dim=1)
            num_correct += (batch_pred == label).sum(dim=0).item()

            if num_item_train * 0.02 <= iter_count:  # Log every 2% of the training data
                for key, value in loss_dict.items():
                    if isinstance(value, AverageMeter):
                        self.logger.wandbLog({key: value.avg})
                    else:
                        self.logger.wandbLog({key: value})
                # self.logger.wandbLog({'Loss': loss_sum / float(iter_count)})
                # loss_sum = 0
                iter_count = 0
        self.logger.wandbLog(
            {'Train Acc': (num_correct / num_total) * 100})

    def test(self, is_dev=False):

        return self._test(self.dev_loader, mode='validation') if is_dev else self._test(self.test_loader)

    def _test(self, test_loader, mode='evaluation'):
        self.model.eval()

        scores = []
        labels = []
        num_correct = 0.0
        iter_count = 0
        num_total = 0.0

        pbar = tqdm(test_loader, mode)
        loss_sum = 0
        iter_count = 0
        with torch.no_grad():
            for utter_id, x, label in pbar:
                iter_count += 1
                x, label = x.to(
                    self.device), label.view(-1).type(torch.int64).to(self.device)
                batch_size = x.size(0)
                num_total += batch_size
                x = self.preprocessor(x)

                x = self.model(x)

                loss = self.loss_fn(x, label)
                _, batch_pred = x.max(dim=1)
                num_correct += (batch_pred == label).sum(dim=0).item()

                loss_sum += (loss.item() * batch_size)

        eval_loss = loss_sum / num_total

        accuracy = (num_correct / num_total) * 100
        self.logger.wandbLog(
            {'Dev Acc': accuracy, 'Dev Loss': eval_loss})
        return eval_loss, accuracy

    def calculate_EER(self, scores, labels):
        # Save the scores and labels

        fpr, tpr, _ = metrics.roc_curve(labels, scores, pos_label=1)
        eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
        return eer * 100
