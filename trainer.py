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
