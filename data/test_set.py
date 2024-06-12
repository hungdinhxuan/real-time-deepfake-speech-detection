from typing import Any
import torch
import torchaudio
import torch.utils.data as data
import os
import random
import config
import numpy as np
import librosa


class ASVspoof2021LA_eval(data.Dataset):

    def __init__(self, sys_config, exp_config, skip_adjustDuration=False):

        self.duration = exp_config.test_duration_sec * exp_config.sample_rate
        self.skip_adjustDuration = skip_adjustDuration

        path_label = sys_config.path_label_asv_spoof_2021_la_eval
        path_eval = sys_config.path_asv_spoof_2021_la_eval

        self.data_list = []
        """This contains tuples like (file_path:str, attack_type, is_real:int)"""

        for line in open(path_label).readlines():
            line = line.replace('\n', '').split(' ')
            # if line[7] != 'eval':
            #     continue

            file, attack_type, label = line[1], line[4], 1 if line[4] == 'bonafide' else 0
            file = os.path.join(path_eval, f'{file}.flac')

            self.data_list.append((file, attack_type, label))

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index: Any) -> Any:

        utter_id, _, label = self.data_list[index]
        utter, _ = torchaudio.load(utter_id)
        utter = self.adjustDuration(utter)
        # Get last name of utter_id
        utter_id = os.path.basename(utter_id)
        utter_id = utter_id.split('.')[0]
        return utter_id, utter, label

    def adjustDuration(self, x):
        """_summary_
        use test data with specific duration from start of the audio 
        Args:
            x (_type_): _description_

        Returns:
            _type_: _description_
        """

        if len(x.shape) == 2:
            x = x.squeeze()

        x_len = len(x)
        if x_len < self.duration:
            tmp = [x for i in range(0, (self.duration // x_len))]

            residue = self.duration % x_len
            if residue > 0:
                tmp.append(x[0:residue])

            x = torch.cat(tmp, dim=0)

        return x[0: self.duration]


class ASVspoof2019LA_eval(data.Dataset):

    def __init__(self, sys_config, exp_config, skip_adjustDuration=False):

        self.duration = exp_config.test_duration_sec * exp_config.sample_rate
        self.skip_adjustDuration = skip_adjustDuration

        path_label = sys_config.path_label_asv_spoof_2019_la_eval

        self.data_list = []
        """This contains tuples like (file_path:str, attack_type, is_real:int)"""

        for line in open(path_label).readlines():

            line = line.replace('\n', '').split(' ')
            file, attack_type, label = line[1], line[3], 1 if line[4] == 'bonafide' else 0

            # Not include no_speech
            if 'no_speech' in file and not exp_config.include_non_speech:
                continue

            # Not include residual
            if 'residual' in file and not exp_config.include_residual:
                continue

            file = os.path.join(
                sys_config.path_asv_spoof_2019_la_eval, f'{file}.flac')

            self.data_list.append((file, attack_type, label))

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index: Any) -> Any:

        utter_id, _, label = self.data_list[index]
        # utter, _ = torchaudio.load(utter_id)
        utter, sr = librosa.load(utter_id, sr=16000)
        utter = torch.from_numpy(utter)
        utter = self.adjustDuration(utter)
        # Get last name of utter_id

        utter_id = os.path.basename(utter_id)
        utter_id = utter_id.split('.')[0]
        return utter_id, utter, label

    def adjustDuration(self, x):
        x = x.squeeze() if len(x.shape) == 2 else x
        x_len = len(x)

        if x_len < self.duration:

            tmp = [x] * (self.duration // x_len)

            residue = self.duration % x_len
            if residue > 0:
                tmp.append(x[:residue])

            x = torch.cat([t if isinstance(t, torch.Tensor)
                          else torch.from_numpy(t) for t in tmp]).float()

        start_seg = random.randint(0, len(x) - self.duration)
        return x[start_seg: start_seg + self.duration]


def pad(x, max_len=64600):
    x_len = x.shape[0]
    if x_len >= max_len:
        return x[:max_len]
    # need to pad
    num_repeats = int(max_len / x_len)+1
    padded_x = np.tile(x, (1, num_repeats))[:, :max_len][0]
    return padded_x


class ASVspoof2021DF_eval(data.Dataset):

    def __init__(self, sys_config, exp_config):
        print(exp_config)
        self.duration = int(exp_config.test_duration_sec *
                            exp_config.sample_rate)
        self.is_random_start = exp_config.is_random_start
        path_label = sys_config.path_label_asv_spoof_2021_df_eval
        path_eval = sys_config.path_asv_spoof_2021_df_eval

        self.data_list = []
        """This contains tuples like (file_path:str, attack_type, is_real:int)"""

        for line in open(path_label).readlines():
            line = line.replace('\n', '').split(' ')
            # if line[7] != 'eval':
            #     continue

            if not sys_config.path_label_asv_spoof_2021_la_eval_spec:
                file, attack_type, label = line[1], line[5], 1 if line[5] == 'bonafide' else 0
            else:
                file, attack_type, label = line[0], '', 1

            file = os.path.join(path_eval, f'{file}.flac')
            self.data_list.append((file, attack_type, label))

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index: Any) -> Any:

        utter_id, _, label = self.data_list[index]

        # Torchaudio load
        utter, _ = torchaudio.load(utter_id)
        # Torchaudio load

        # Librosa load
        # utter, sr = librosa.load(utter_id, sr=16000)
        # #utter = pad(utter, self.duration)
        # utter = torch.from_numpy(utter)
        # Librosa load

        utter = self.adjustDuration_random_start(
            utter) if self.is_random_start else self.adjustDuration(utter)

        # Get last name of utter_id
        utter_id = os.path.basename(utter_id)
        utter_id = utter_id.split('.')[0]

        return utter_id, utter, label

    def adjustDuration(self, x):
        """_summary_
        use test data with specific duration from start of the audio 
        Args:
            x (_type_): _description_

        Returns:
            _type_: _description_
        """

        if len(x.shape) == 2:
            x = x.squeeze()

        x_len = len(x)
        if x_len < self.duration:
            # repeat x to fill the duration
            tmp = [x for i in range(0, (self.duration // x_len))]

            # add residue if any left to fill the duration
            residue = self.duration % x_len
            if residue > 0:
                tmp.append(x[0:residue])

            # concatenate all the repeated x
            x = torch.cat(tmp, dim=0)

        return x[0: self.duration]  # first duration seconds

    def adjustDuration_random_start(self, x):

        if len(x.shape) == 2:
            x = x.squeeze()
        x_len = len(x)

        if x_len < self.duration:
            tmp = [x for i in range(0, (self.duration // x_len))]

            residue = self.duration % x_len
            if residue > 0:
                tmp.append(x[0:residue])

            x = torch.cat(tmp, dim=0)

        start_seg = random.randint(0, len(x) - self.duration)
        end_seg = start_seg + self.duration
        # print(f"start_seg: {start_seg}, end_seg: {end_seg}")

        return x[start_seg: end_seg]


class InTheWild(data.Dataset):

    def __init__(self, sys_config, exp_config, skip_adjustDuration=False):

        self.duration = int(exp_config.test_duration_sec *
                            exp_config.sample_rate)
        self.is_random_start = exp_config.is_random_start

        self.skip_adjustDuration = skip_adjustDuration

        path_label = sys_config.path_label_in_the_wild

        self.data_list = []
        """This contains tuples like (file_path:str, attack_type, is_real:int)"""

        for line in open(path_label).readlines():

            line = line.strip().split()
            file,  label = line[0], 1 if line[1] == 'bonafide' else 0

            if not file.endswith('.wav'):
                file = os.path.join(
                    sys_config.path_in_the_wild, f'{file}.wav')
            else:
                file = os.path.join(
                    sys_config.path_in_the_wild, f'{file}')

            self.data_list.append((file, None, label))

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index: Any) -> Any:

        utter_id, _, label = self.data_list[index]
        utter, _ = torchaudio.load(utter_id)
        utter = self.adjustDuration_random_start(
            utter) if self.is_random_start else self.adjustDuration(utter)
        # Get last name of utter_id
        utter_id = os.path.basename(utter_id)
        utter_id = utter_id.split('.')[0]
        return utter_id, utter, label

    def adjustDuration(self, x):
        """_summary_
        use test data with specific duration from start of the audio 
        Args:
            x (_type_): _description_

        Returns:
            _type_: _description_
        """

        if len(x.shape) == 2:
            x = x.squeeze()

        x_len = len(x)
        if x_len < self.duration:
            # repeat x to fill the duration
            tmp = [x for i in range(0, (self.duration // x_len))]

            # add residue if any left to fill the duration
            residue = self.duration % x_len
            if residue > 0:
                tmp.append(x[0:residue])

            # concatenate all the repeated x
            x = torch.cat(tmp, dim=0)

        return x[0: self.duration]  # first duration seconds

    def adjustDuration_random_start(self, x):

        if len(x.shape) == 2:
            x = x.squeeze()
        x_len = len(x)

        if x_len < self.duration:
            tmp = [x for i in range(0, (self.duration // x_len))]

            residue = self.duration % x_len
            if residue > 0:
                tmp.append(x[0:residue])

            x = torch.cat(tmp, dim=0)

        start_seg = random.randint(0, len(x) - self.duration)
        end_seg = start_seg + self.duration
        # print(f"start_seg: {start_seg}, end_seg: {end_seg}")

        return x[start_seg: end_seg]


class FakeOrReal(data.Dataset):

    def __init__(self, sys_config, exp_config, skip_adjustDuration=False):

        self.duration = int(exp_config.test_duration_sec *
                            exp_config.sample_rate)
        self.is_random_start = exp_config.is_random_start

        self.skip_adjustDuration = skip_adjustDuration

        path_label = sys_config.path_label_fake_or_real

        self.data_list = []
        """This contains tuples like (file_path:str, attack_type, is_real:int)"""

        for line in open(path_label).readlines():

            line = line.strip().split()
            file, subset, label = line[0], line[1], 1 if line[2] == 'bonafide' else 0

            if not file.endswith('.wav'):
                file = os.path.join(
                    sys_config.path_fake_or_real, f'{file}.wav')
            else:
                file = os.path.join(
                    sys_config.path_fake_or_real, f'{file}')

            self.data_list.append((file, None, label))

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index: Any) -> Any:

        utter_id, _, label = self.data_list[index]
        utter, _ = torchaudio.load(utter_id)
        utter = self.adjustDuration_random_start(
            utter) if self.is_random_start else self.adjustDuration(utter)
        # Get last name of utter_id
        # utter_id = os.path.basename(utter_id)
        # utter_id = utter_id.split('.')[0]
        return utter_id, utter, label

    def adjustDuration(self, x):
        """_summary_
        use test data with specific duration from start of the audio 
        Args:
            x (_type_): _description_

        Returns:
            _type_: _description_
        """

        if len(x.shape) == 2:
            x = x.squeeze()

        x_len = len(x)
        if x_len < self.duration:
            # repeat x to fill the duration
            tmp = [x for i in range(0, (self.duration // x_len))]

            # add residue if any left to fill the duration
            residue = self.duration % x_len
            if residue > 0:
                tmp.append(x[0:residue])

            # concatenate all the repeated x
            x = torch.cat(tmp, dim=0)

        return x[0: self.duration]  # first duration seconds

    def adjustDuration_random_start(self, x):

        if len(x.shape) == 2:
            x = x.squeeze()
        x_len = len(x)

        if x_len < self.duration:
            tmp = [x for i in range(0, (self.duration // x_len))]

            residue = self.duration % x_len
            if residue > 0:
                tmp.append(x[0:residue])

            x = torch.cat(tmp, dim=0)

        start_seg = random.randint(0, len(x) - self.duration)
        end_seg = start_seg + self.duration
        # print(f"start_seg: {start_seg}, end_seg: {end_seg}")

        return x[start_seg: end_seg]

class ASVSpoof5(data.Dataset):

    def __init__(self, sys_config, exp_config, skip_adjustDuration=False):

        self.duration = int(exp_config.test_duration_sec *
                            exp_config.sample_rate)
        self.is_random_start = exp_config.is_random_start

        self.skip_adjustDuration = skip_adjustDuration

        path_label = sys_config.path_label_asvspoof5

        self.data_list = []
        """This contains tuples like (file_path:str, attack_type, is_real:int)"""

        for line in open(path_label).readlines():

            line = line.strip().split()
            file, subset, label = line[0], line[1], 1 if line[2] == 'bonafide' else 0

            file = os.path.join(
                    sys_config.path_asvspoof5, f'{file}')

            self.data_list.append((file, None, label))

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index: Any) -> Any:

        utter_id, _, label = self.data_list[index]
        utter, _ = torchaudio.load(utter_id)
        utter = self.adjustDuration_random_start(
            utter) if self.is_random_start else self.adjustDuration(utter)
        # Get last name of utter_id
        # utter_id = os.path.basename(utter_id)
        # utter_id = utter_id.split('.')[0]
        return utter_id, utter, label

    def adjustDuration(self, x):
        """_summary_
        use test data with specific duration from start of the audio 
        Args:
            x (_type_): _description_

        Returns:
            _type_: _description_
        """

        if len(x.shape) == 2:
            x = x.squeeze()

        x_len = len(x)
        if x_len < self.duration:
            # repeat x to fill the duration
            tmp = [x for i in range(0, (self.duration // x_len))]

            # add residue if any left to fill the duration
            residue = self.duration % x_len
            if residue > 0:
                tmp.append(x[0:residue])

            # concatenate all the repeated x
            x = torch.cat(tmp, dim=0)

        return x[0: self.duration]  # first duration seconds

    def adjustDuration_random_start(self, x):

        if len(x.shape) == 2:
            x = x.squeeze()
        x_len = len(x)

        if x_len < self.duration:
            tmp = [x for i in range(0, (self.duration // x_len))]

            residue = self.duration % x_len
            if residue > 0:
                tmp.append(x[0:residue])

            x = torch.cat(tmp, dim=0)

        start_seg = random.randint(0, len(x) - self.duration)
        end_seg = start_seg + self.duration
        # print(f"start_seg: {start_seg}, end_seg: {end_seg}")

        return x[start_seg: end_seg]