import torch
import torch.nn as nn
from conformer import ConformerBlock
from torch.nn.modules.transformer import _get_clones
from .fe import *


class MyConformer(nn.Module):
    def __init__(self, emb_size=128, heads=4, ffmult=4, exp_fac=2, kernel_size=16, n_encoders=1):
        super(MyConformer, self).__init__()
        self.dim_head = int(emb_size/heads)
        self.dim = emb_size
        self.heads = heads
        self.kernel_size = kernel_size
        self.n_encoders = n_encoders
        self.encoder_blocks = _get_clones(ConformerBlock(dim=emb_size, dim_head=self.dim_head, heads=heads,
                                                         ff_mult=ffmult, conv_expansion_factor=exp_fac, conv_kernel_size=kernel_size),
                                          n_encoders)
        self.class_token = nn.Parameter(torch.rand(1, emb_size))
        self.fc5 = nn.Linear(emb_size, 2)

    def forward(self, x, device):  # x shape [bs, tiempo, frecuencia]
        x = torch.stack([torch.vstack((self.class_token, x[i]))
                        for i in range(len(x))])  # [bs,1+tiempo,emb_size]
        for layer in self.encoder_blocks:
            x = layer(x)  # [bs,1+tiempo,emb_size]
        embedding = x[:, 0, :]  # [bs, emb_size]
        out = self.fc5(embedding)  # [bs,2]
        return out, embedding


class Model(nn.Module):
    def __init__(self, device, ssl_cpkt_path, **kwargs):
        super().__init__()
        self.device = device
        ##
        # Default config from conformer
        ##
        emb_size = kwargs.get('emb_size', 144)
        heads = kwargs.get('heads', 4)
        kernel_size = kwargs.get('kernel_size', 31)
        n_encoders = kwargs.get('n_encoders', 4)
        ####
        # create network wav2vec 2.0
        ####
        self.ssl_model = XLSR_FE(device)
        self.LL = nn.Linear(1024, emb_size)
        print('W2V + Conformer')
        self.first_bn = nn.BatchNorm2d(num_features=1)
        self.selu = nn.SELU(inplace=True)

        self.conformer = MyConformer(emb_size=emb_size, n_encoders=n_encoders,
                                     heads=heads, kernel_size=kernel_size)

    def forward(self, x):
        # -------pre-trained Wav2vec model fine tunning ------------------------##
        x_ssl_feat = self.ssl_model.extract_feat(x.squeeze(-1))
        # (bs,frame_number,feat_out_dim) (bs, 208, 256)
        x = self.LL(x_ssl_feat)
        x = x.unsqueeze(dim=1)  # add channel #(bs, 1, frame_number, 256)
        x = self.first_bn(x)
        x = self.selu(x)
        x = x.squeeze(dim=1)
        out, _ = self.conformer(x, self.device)
        return out



class MyModel(nn.Module):
    def __init__(self, device, ssl_cpkt_path, **kwargs):
        super().__init__()
        self.device = device
        ##
        # Default config from conformer
        ##
        emb_size = kwargs.get('emb_size', 144)
        heads = kwargs.get('heads', 4)
        kernel_size = kwargs.get('kernel_size', 31)
        n_encoders = kwargs.get('n_encoders', 4)
        ####
        # create network wav2vec 2.0
        ####
        self.ssl_model = My_XLSR_FE(device, **kwargs)
        self.LL = nn.Linear(1024, emb_size)
        print('W2V + Conformer')
        self.first_bn = nn.BatchNorm2d(num_features=1)
        self.selu = nn.SELU(inplace=True)

        self.conformer = MyConformer(emb_size=emb_size, n_encoders=n_encoders,
                                     heads=heads, kernel_size=kernel_size)

    def forward(self, x):
        # -------pre-trained Wav2vec model fine tunning ------------------------##
        x_ssl_feat = self.ssl_model.extract_feat(x.squeeze(-1))
        # (bs,frame_number,feat_out_dim) (bs, 208, 256)
        x = self.LL(x_ssl_feat)
        x = x.unsqueeze(dim=1)  # add channel #(bs, 1, frame_number, 256)
        x = self.first_bn(x)
        x = self.selu(x)
        x = x.squeeze(dim=1)
        out, _ = self.conformer(x, self.device)
        return out



# class Model2(nn.Module):  # Variable len
#     def __init__(self, args, device):
#         super().__init__()
#         self.device = device
#         self.ssl_model = SSLModel(self.device)
#         self.LL = nn.Linear(1024, args.emb_size)
#         print('W2V + Conformer: Variable Length')
#         self.first_bn = nn.BatchNorm2d(num_features=1)
#         self.selu = nn.SELU(inplace=True)
#         self.conformer = MyConformer(emb_size=args.emb_size, n_encoders=args.num_encoders,
#                                      heads=args.heads, kernel_size=args.kernel_size)

#     def forward(self, x):  # x is a list of np arrays
#         nUtterances = len(x)
#         output = torch.zeros(nUtterances, 2).to(self.device)
#         for n, feat in enumerate(x):
#             input_x = torch.from_numpy(feat[:, :]).float().to(self.device)
#             x_ssl_feat = self.ssl_model.extract_feat(input_x.squeeze(-1))
#             f = self.LL(x_ssl_feat)
#             f = f.unsqueeze(dim=1)
#             f = self.first_bn(f)
#             f = self.selu(f)
#             f = f.squeeze(dim=1)
#             out, _ = self.conformer(f, self.device)
#             output[n, :] = out
#         return output
