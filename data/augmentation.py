import torch
import torch.nn as nn
import argparse
from torch_audiomentations import Compose, AddColoredNoise, HighPassFilter, LowPassFilter, Gain
from data.RawBoost import ISD_additive_noise, LnL_convolutive_noise, SSI_additive_noise, normWav
import audiomentations as aa
import os

# AUDIOSET_DIR = "/datad/utils/my-audioset-processing/output"
# AUDIOSET_AUGMENTED_DIR = [os.path.join(
#     AUDIOSET_DIR, dir
# ) for dir in os.listdir(
#     AUDIOSET_DIR) if os.path.isdir(os.path.join(AUDIOSET_DIR, dir)) and "synthe" not in dir and "Syn" not in dir]


class WaveformAugmetation(nn.Module):
    def __init__(self, aug_list=['ACN', 'HPF', 'LPF', 'GAN', 'RawBoost1', 'RawBoost2', 'RawBoost3', 'RawBoost4', 'RawBoost5', 'RawBoost6', 'RawBoost7', 'RawBoost8'],
                 params={
                     'sr': 16000,
                     'ACN': {
                         'min_snr_in_db': 10, 'max_snr_in_db': 40, 'min_f_decay': -2.0, 'max_f_decay': 2.0, 'p': 0.5
                     },
                     'HPF': {
                         'min_cutoff_freq': 20.0, 'max_cutoff_freq': 2400.0, 'p': 0.5
                     },
                     'LPF': {
                         'min_cutoff_freq': 150.0, 'max_cutoff_freq': 7500.0, 'p': 0.5
                     },
                     'GAN': {
                         'min_gain_in_db': -12.0, 'max_gain_in_db': 12.0, 'p': 0.75
                     },

    }
    ):
        # RawBoost option = (min_snr_in_db=10, max_snr_in_db=40)
        # torch_audiomentations option = (min_snr_in_db=3, max_snr_in_db=30)
        # torch_audiomentations option = (min_gain_in_db = -18.0, max_gain_in_db = 6.0)

        super(WaveformAugmetation, self).__init__()
        self.sr = params['sr']
        transforms = []
        if 'ACN' in aug_list:
            transforms.append(
                AddColoredNoise(
                    min_snr_in_db=params['ACN']['min_snr_in_db'],
                    max_snr_in_db=params['ACN']['max_snr_in_db'],
                    min_f_decay=params['ACN']['min_f_decay'],
                    max_f_decay=params['ACN']['max_f_decay'],
                    p=params['ACN']['p'],
                )
            )
        if 'HPF' in aug_list:
            transforms.append(
                HighPassFilter(
                    min_cutoff_freq=params['HPF']['min_cutoff_freq'],
                    max_cutoff_freq=params['HPF']['max_cutoff_freq'],
                    p=params['HPF']['p'],
                )
            )
        if 'LPF' in aug_list:
            transforms.append(
                LowPassFilter(
                    min_cutoff_freq=params['LPF']['min_cutoff_freq'],
                    max_cutoff_freq=params['LPF']['max_cutoff_freq'],
                    p=params['LPF']['p'],
                )
            )
        if 'GAN' in aug_list:
            transforms.append(
                Gain(
                    min_gain_in_db=params['GAN']['min_gain_in_db'],
                    max_gain_in_db=params['GAN']['max_gain_in_db'],
                    p=params['GAN']['p'],
                )
            )
        self.aug_list = aug_list
        self.apply_augmentation = Compose(transforms)
        self.device = 'cpu'

    def forward(self, wav):
        # device sync
        if wav.device != self.device:
            self.device = wav.device
            self.apply_augmentation.to(wav.device)
        # data.shape: 1 dimention (WaveSize)
        augmented_wav = self.apply_augmentation(
            wav.unsqueeze(1), self.sr).squeeze(1)

        return augmented_wav


def process_audiomentations(feature, sr):
    """ DA using audiomentations library    
    """
    # aa.ApplyImpulseResponse(ir_path="/path/to/sound_folder", p=1.0)

    augment = aa.Compose([
        aa.AddBackgroundNoise(
            sounds_path="/home/hungdx/audioset_mix_30_percent_trimmed", p=0.75),
        aa.AdjustDuration(duration_seconds=4, p=1.0, padding_mode="wrap"),
        aa.TimeStretch(min_rate=0.8, max_rate=1.2,
                       leave_length_unchanged=True, p=0.75),
        aa.Gain(min_gain_in_db=-12, max_gain_in_db=12, p=0.75),
        aa.AirAbsorption(min_distance=1.0, max_distance=20.0, p=0.75),
        aa.TimeMask(min_band_part=0.1, max_band_part=0.15, fade=True, p=0.5),
        aa.Mp3Compression(min_bitrate=96, max_bitrate=320, p=0.3),
    ])
    return augment(samples=feature, sample_rate=sr)
