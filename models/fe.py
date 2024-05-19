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


class Wrapper_XLSR_FE(nn.Module):

    def __init__(self, device, **kwargs):
        super().__init__()
        self.num_layers = kwargs.get('num_layers', 24)
        if self.num_layers < 1 or self.num_layers > 24:
            raise ValueError(
                "Number of layers must be at least 1 and at most 24.")
        self.xlsr = XLSR_FE(device)
        self.out_dim = self.xlsr.out_dim

        # Remove transformer layers from the model
        self.xlsr.model.encoder.layers = self.xlsr.model.encoder.layers[:self.num_layers]
        # self.layer_norm = fairseq.modules.LayerNorm(self.out_dim)
        self.model_wrapped = Inspect(self.xlsr.model, layer=[
                                     f'encoder.layers.{i}.final_layer_norm' for i in range(self.num_layers)])
        del self.xlsr
        # # Start with a minimal tensor
        # self.w = nn.Parameter(torch.randn((
        #     1, 1), device=device), requires_grad=True).to(device)

    def forward(self, x):
        _, middle_layers = self.model_wrapped(x)

        # Sum all the layers
        # (feat, batch, out_dim)
        sum_tensor = torch.sum(torch.stack(middle_layers), dim=0)

        # Reshape to (batch, feat, out_dim)
        # sum_tensor = sum_tensor.permute(1, 0, 2)
        sum_tensor = sum_tensor.transpose(0, 1)

        # Pass sum_tensor through a batch_norm layer
        # sum_tensor_layer_norm = self.layer_norm(sum_tensor)

        return sum_tensor

    def extract_feat(self, x):
        return self.forward(x)


class My_XLSR_FE(nn.Module):

    def __init__(self, device, **kwargs):
        super().__init__()
        self.num_layers = kwargs.get('num_layers', 24)
        if self.num_layers < 1 or self.num_layers > 24:
            raise ValueError(
                "Number of layers must be at least 1 and at most 24.")
        model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([
                                                                                 '/datad/hungdx/Rawformer-implementation-anti-spoofing/pretrained/xlsr2_300m.pt'])
        self.model = model[0]
        self.model = self.model.to(device)
        self.out_dim = 1024

        # Remove transformer layers from the model
        self.model.encoder.layers = self.model.encoder.layers[:self.num_layers]

        # # Start with a minimal tensor
        # self.w = nn.Parameter(torch.randn((
        #     1, 1), device=device), requires_grad=True).to(device)

    def forward(self, x):
        input_tmp = x[:, :, 0] if x.ndim == 3 else x
        layer_results = self.model(input_tmp, mask=False, features_only=True)[
            'layer_results']

        # extract the representation of each layer
        layer_reps = [layer[-1] for layer in layer_results]

        # Sum all the layers
        # (feat, batch, out_dim)
        sum_tensor = torch.sum(torch.stack(layer_reps), dim=0)

        # Reshape to (batch, feat, out_dim)
        sum_tensor = sum_tensor.transpose(0, 1)

        return sum_tensor

    def extract_feat(self, x):
        return self.forward(x)
