from .aasist_modules import *
from torch import nn
import torch
import torch
import fairseq


class XLSR_FE(nn.Module):
    def __init__(self, device):
        super().__init__()
        model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([
                                                                                 '/datad/hungdx/Rawformer-implementation-anti-spoofing/pretrained/xlsr2_300m.pt'])
        self.model = model[0]
        self.model = self.model.to(device)
        self.out_dim = 1024

    def extract_feat(self, input_data):
        input_tmp = input_data[:, :, 0] if input_data.ndim == 3 else input_data
        emb = self.model(input_tmp, mask=False, features_only=True)[
            'x']
        return emb

    def forward(self, input_data):
        return self.extract_feat(input_data)

    def partial_freeze_layers(self, target_layers: list, non_target_layers: list):
        # Freeze all layers except the non_target_layers layers
        for name, param in self.model.named_parameters():
            if any([layer in name for layer in target_layers]) and not any([layer in name for layer in non_target_layers]):
                print(f"Freezing layer: {name}")
                param.requires_grad = False

        # Randomly initialize the non_target_layers layers
        self.random_init_layers(non_target_layers)

    def random_init_layers(self, target_layers: list):
        for name, param in self.model.named_parameters():
            if any([layer in name for layer in target_layers]) and param.dim() >= 2:
                print(f"Randomly initializing layer: {name}")
                torch.nn.init.xavier_uniform_(param)


def middle_indices(array_length, number_of_middle_elements):
    # Calculate the start index
    start_index = (array_length - number_of_middle_elements) // 2
    # Calculate the end index
    end_index = start_index + number_of_middle_elements
    # Create a list of the middle indices
    middle_indices = list(range(start_index, end_index))
    return middle_indices


class My_XLSR_FE(nn.Module):

    def __init__(self, device, **kwargs):
        super().__init__()
        self.num_layers = kwargs.get('num_layers', 24)
        self.order = kwargs.get('order', 'first')
        self.custom_order = kwargs.get('custom_order', None)
        if self.num_layers < 1 or self.num_layers > 24:
            raise ValueError(
                "Number of layers must be at least 1 and at most 24.")
        model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([
                                                                                 '/datad/hungdx/Towards-Real-Time-Deepfake-Speech-Detection-in-Resource-Limited-Scenarios/pretrained/xlsr2_300m.pt'])
        self.model = model[0]
        self.model = self.model.to(device)
        self.out_dim = 1024

        if self.order == 'last':
            # Get the last n layers
            self.model.encoder.layers = self.model.encoder.layers[-self.num_layers:]
        elif self.order == 'first':
            # Get the first n layers
            self.model.encoder.layers = self.model.encoder.layers[:self.num_layers]
        elif self.order == 'middle':
            indices = middle_indices(24, self.num_layers)

            self.model.encoder.layers = nn.ModuleList([
                self.model.encoder.layers[i] for i in indices])
        else:
            if self.custom_order is None:
                raise ValueError(
                    "Custom order must be provided as a list of integers (0-23).")

            # Check if the custom order is valid
            if type(self.custom_order) != list:
                raise ValueError("Custom order must be a list of integers.")

            self.model.encoder.layers = nn.ModuleList([
                self.model.encoder.layers[i] for i in self.custom_order])

    def forward(self, x):
        return self.extract_feat(x)

    def extract_feat(self, x):
        input_tmp = x[:, :, 0] if x.ndim == 3 else x
        emb = self.model(input_tmp, mask=False, features_only=True)[
            'x']
        return emb
