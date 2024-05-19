from typing import Any
import torch
import torchaudio
import torch.utils.data as data
import os
import random
from data.augmentation import WaveformAugmetation, process_audiomentations
import config
from data.RawBoost import process_Rawboost_feature
import librosa
from torch.multiprocessing import Pool


class Args:
    def __init__(self, dictionary):
        for key, value in dictionary.items():
            setattr(self, key, value)


def process_file(path_label, path_asv_spoof, exp_config):
    data_list = []
    num_of_spoof = 0
    num_of_bonafide = 0
    for line in open(path_label).readlines():
        line = line.replace('\n', '').split(' ')
        file, attack_type, label = line[1], line[3], 1 if line[4] == 'bonafide' else 0

        if label == 1:
            num_of_bonafide += 1
        else:
            num_of_spoof += 1 
        # This part is for my own experiment
        # Not include no_speech
        if 'no_speech' in file and not exp_config.include_non_speech:
            continue

        # Not include residual
        if 'residual' in file and not exp_config.include_residual:
            continue
        # This part is for my own experiment

        file = os.path.join(path_asv_spoof, f'{file}.flac')
        data_list.append((file, attack_type, label))
    return data_list, num_of_spoof, num_of_bonafide


class ASVspoof2019LA(data.Dataset):
    """_summary_


    """

    def __init__(self, sys_config, exp_config, is_train=True):
        super(ASVspoof2019LA, self).__init__()
        self.sample_rate = exp_config.sample_rate
        self.duration = int(
            exp_config.train_duration_sec * exp_config.sample_rate)
        self.is_train = is_train
        path_label_train = sys_config.path_label_asv_spoof_2019_la_train
        path_label_dev = sys_config.path_label_asv_spoof_2019_la_dev

        ## ===================================================Rawboost data augmentation ======================================================================#

        # LnL_convolutive_noise parameters
        args = {
            'nBands': 5,
            'minF': 20,
            'maxF': 8000,
            'minBW': 100,
            'maxBW': 1000,
            'minCoeff': 10,
            'maxCoeff': 100,
            'minG': 0,
            'maxG': 0,
            'minBiasLinNonLin': 5,
            'maxBiasLinNonLin': 20,
            'N_f': 5,
            'P': 10,
            'g_sd': 2,
            'SNRmin': 10,
            'SNRmax': 40
        }
        self.args = Args(args)
        self.data_augmentation_list = exp_config.data_augmentation
        
        ## ===================================================Rawboost data augmentation ======================================================================#

        self.data_list = []
        """This contains tuples like (file_path:str, attack_type, is_real:int)"""

        # ------------------- save train list ------------------- #
        if is_train:
            self.data_list, num_of_spoof, num_of_bonafide = process_file(
                path_label_train, sys_config.path_asv_spoof_2019_la_train, exp_config)
        else:
            self.data_list, num_of_spoof, num_of_bonafide = process_file(
                path_label_dev, sys_config.path_asv_spoof_2019_la_dev, exp_config)
        
        self.num_of_spoof = num_of_spoof
        self.num_of_bonafide = num_of_bonafide

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index: Any) -> Any:
        utter, _, label = self.data_list[index]
        # utter, _ = librosa.load(utter, sr=self.sample_rate)
        utter, _ = torchaudio.load(utter)
        utter = utter.squeeze()

        if self.is_train:
            # =============================== Data Augmentation =============================== #
            algo = next(
                (i for i in range(1, 9) if f"RawBoost{i}" in self.data_augmentation_list), -1)

            if algo != -1:
                # Convert to numpy array
                utter = utter.numpy()
                utter = process_Rawboost_feature(
                    utter, self.sample_rate, self.args, algo)
                utter = torch.from_numpy(utter).float()
            elif "mul_augment" in self.data_augmentation_list:
                utter = process_audiomentations(utter, self.sample_rate)
            # =============================== Data Augmentation =============================== #

        utter = self.adjustDuration(utter)
        if not isinstance(utter, torch.Tensor):
            utter = torch.from_numpy(utter).float()
        return _, utter, label

    def adjustDuration(self, x):
        # x = x.squeeze() if len(x.shape) == 2 else x
        x_len = len(x)

        if x_len < self.duration:
            if "randomly_mul_augment_padding" in self.data_augmentation_list and self.is_train:
                # print("randomly_mul_augment_padding")
                tmp = [process_audiomentations(
                    x, self.sample_rate) for _ in range(self.duration // x_len)]
            else:
                tmp = [x] * (self.duration // x_len)

            residue = self.duration % x_len
            if residue > 0:
                tmp.append(x[:residue])

            # x = torch.cat([torch.from_numpy(t) for t in tmp])
            x = torch.cat([t if isinstance(t, torch.Tensor)
                          else torch.from_numpy(t) for t in tmp]).float()

        start_seg = random.randint(0, len(x) - self.duration)

        return x[start_seg: start_seg + self.duration]
