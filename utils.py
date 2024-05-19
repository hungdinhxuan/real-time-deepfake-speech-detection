import numpy as np
import random
import socket
import logging
import os
import torch
import torch.nn.functional as F
from collections import OrderedDict
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def f_state_dict_wrapper(state_dict, data_parallel=False):
    """ a wrapper to take care of state_dict when using DataParallism

    f_model_load_wrapper(state_dict, data_parallel):
    state_dict: pytorch state_dict
    data_parallel: whether DataParallel is used
    
    https://discuss.pytorch.org/t/solved-keyerror-unexpected-
    key-module-encoder-embedding-weight-in-state-dict/1686/3
    """
    if data_parallel is True:
        # if data_parallel is used
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if not k.startswith('module'):
                # if key is not starting with module, add it
                name = 'module.' + k
            else:
                name = k
            new_state_dict[name] = v
        return new_state_dict
    else:
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if not k.startswith('module'):
                name = k
            else:
                # remove module.
                name = k[7:] 
            new_state_dict[name] = v
        return new_state_dict

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0, model_save_path=None):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_eer = None
        self.early_stop = False
        self.eer_min = 100
        self.delta = delta
        self.model_save_path = model_save_path
        self.best_epoch = None

    def __call__(self, eer, model, epoch):

        if self.best_eer is None:
            self.best_eer = eer
            self.save_checkpoint(eer, model, epoch)
        elif eer < self.best_eer + self.delta:
            self.counter += 1
            logger.info('EarlyStopping counter: %s out of %s - Current best eer: %s ',
                        self.counter, self.patience, self.best_eer)
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_eer = eer
            self.save_checkpoint(eer, model, epoch)
            self.counter = 0

    def save_checkpoint(self, eer, model, epoch):
        '''Saves model when eer validation decrease.'''
        if self.verbose:
            logger.info(
                f'eer validation decreased ({self.eer_min:.6f} --> {eer:.6f}).  Saving model ...')
        torch.save(model.state_dict(), os.path.join(
            self.model_save_path, 'best_checkpoint_{}.pth'.format(epoch)))
        # Remove previous best model to save memory
        if epoch > 0:
            previous_best_model_path = os.path.join(
                self.model_save_path, 'best_checkpoint_{}.pth'.format(epoch-1))
            if os.path.exists(previous_best_model_path):
                os.remove(previous_best_model_path)
                logger.debug(
                    f'Removed previous best model at {previous_best_model_path}')

        self.eer_min = eer


# Dataset return sample = (utterance, target, nameFile) #shape of utterance [1, lenAudio]
def my_collate(batch):
    data = [dp[0] for dp in batch]
    label = [dp[1] for dp in batch]
    nameFile = [dp[2] for dp in batch]
    return (data, label, nameFile)


def find_available_port(start_port, end_port):
    for port in range(start_port, end_port):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            result = sock.connect_ex(('localhost', port))
            if result != 0:  # Port is available
                return str(port)
    return None
