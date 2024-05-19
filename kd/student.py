from torch import nn
import torch
import torch.nn.functional as F
from models import *
from torch.quantization import QuantStub, DeQuantStub
from torchdistill.models.registry import register_model
from wav2vec2_linear_nll_multi import BackEnd
from wav2vec2_vib import BackEnd as BackEndVIB
from wav2vec2_vib import VIB
from conformer import ConformerBlock
from torch.nn.modules.transformer import _get_clones
from torchaudio.models import wav2vec2_model


@register_model(key='W2V2BASE_HF_AASISTL')
class W2V2BASE_HF_AASISTL(nn.Module):
    def __init__(self, device):
        super().__init__()
        # AASIST parameters
        filts = [128, [1, 32], [32, 32], [32, 24], [24, 24]]
        gat_dims = [24, 32]
        pool_ratios = [0.4, 0.5, 0.7, 0.5]
        temperatures = [2.0, 2.0, 100.0, 100.0]

        ####
        # create network wav2vec 2.0
        ####
        self.ssl_model = SSLHuggingFaceModel(device=device).to(device)
        self.LL = nn.Linear(self.ssl_model.out_dim, 128)
        self.first_bn = nn.BatchNorm2d(num_features=1)
        self.first_bn1 = nn.BatchNorm2d(num_features=24)
        self.drop = nn.Dropout(0.5, inplace=True)
        self.drop_way = nn.Dropout(0.2, inplace=True)
        self.selu = nn.SELU(inplace=True)

        # RawNet2 encoder
        self.encoder = nn.Sequential(
            nn.Sequential(Residual_block(nb_filts=filts[1], first=True)),
            nn.Sequential(Residual_block(nb_filts=filts[2])),
            nn.Sequential(Residual_block(nb_filts=filts[3])),
            nn.Sequential(Residual_block(nb_filts=filts[4])),
            nn.Sequential(Residual_block(nb_filts=filts[4])),
            nn.Sequential(Residual_block(nb_filts=filts[4])))

        self.attention = nn.Sequential(
            nn.Conv2d(24, 128, kernel_size=(1, 1)),
            nn.SELU(inplace=True),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 24, kernel_size=(1, 1)),

        )
        # position encoding
        self.pos_S = nn.Parameter(torch.randn(1, 42, filts[-1][-1]))

        self.master1 = nn.Parameter(torch.randn(1, 1, gat_dims[0]))
        self.master2 = nn.Parameter(torch.randn(1, 1, gat_dims[0]))

        # Graph module
        self.GAT_layer_S = GraphAttentionLayer(filts[-1][-1],
                                               gat_dims[0],
                                               temperature=temperatures[0])
        self.GAT_layer_T = GraphAttentionLayer(filts[-1][-1],
                                               gat_dims[0],
                                               temperature=temperatures[1])
        # HS-GAL layer
        self.HtrgGAT_layer_ST11 = HtrgGraphAttentionLayer(
            gat_dims[0], gat_dims[1], temperature=temperatures[2])
        self.HtrgGAT_layer_ST12 = HtrgGraphAttentionLayer(
            gat_dims[1], gat_dims[1], temperature=temperatures[2])
        self.HtrgGAT_layer_ST21 = HtrgGraphAttentionLayer(
            gat_dims[0], gat_dims[1], temperature=temperatures[2])
        self.HtrgGAT_layer_ST22 = HtrgGraphAttentionLayer(
            gat_dims[1], gat_dims[1], temperature=temperatures[2])

        # Graph pooling layers
        self.pool_S = GraphPool(pool_ratios[0], gat_dims[0], 0.3)
        self.pool_T = GraphPool(pool_ratios[1], gat_dims[0], 0.3)
        self.pool_hS1 = GraphPool(pool_ratios[2], gat_dims[1], 0.3)
        self.pool_hT1 = GraphPool(pool_ratios[2], gat_dims[1], 0.3)

        self.pool_hS2 = GraphPool(pool_ratios[2], gat_dims[1], 0.3)
        self.pool_hT2 = GraphPool(pool_ratios[2], gat_dims[1], 0.3)

        self.out_layer = nn.Linear(5 * gat_dims[1], 2)

    def forward(self, x):
        # -------pre-trained Wav2vec model fine tunning ------------------------##
        x_ssl_feat = self.ssl_model(x.squeeze(-1))
        x = self.LL(x_ssl_feat)  # (bs,frame_number,feat_out_dim)

        # post-processing on front-end features
        x = x.transpose(1, 2)  # (bs,feat_out_dim,frame_number)
        x = x.unsqueeze(dim=1)  # add channel
        x = F.max_pool2d(x, (3, 3))
        x = self.first_bn(x)
        x = self.selu(x)

        # RawNet2-based encoder
        x = self.encoder(x)
        x = self.first_bn1(x)
        x = self.selu(x)

        w = self.attention(x)

        # ------------SA for spectral feature-------------#
        w1 = F.softmax(w, dim=-1)
        m = torch.sum(x * w1, dim=-1)
        e_S = m.transpose(1, 2) + self.pos_S

        # graph module layer
        gat_S = self.GAT_layer_S(e_S)
        out_S = self.pool_S(gat_S)  # (#bs, #node, #dim)

        # ------------SA for temporal feature-------------#
        w2 = F.softmax(w, dim=-2)
        m1 = torch.sum(x * w2, dim=-2)

        e_T = m1.transpose(1, 2)

        # graph module layer
        gat_T = self.GAT_layer_T(e_T)
        out_T = self.pool_T(gat_T)

        # learnable master node
        master1 = self.master1.expand(x.size(0), -1, -1)
        master2 = self.master2.expand(x.size(0), -1, -1)

        # inference 1
        out_T1, out_S1, master1 = self.HtrgGAT_layer_ST11(
            out_T, out_S, master=self.master1)

        out_S1 = self.pool_hS1(out_S1)
        out_T1 = self.pool_hT1(out_T1)

        out_T_aug, out_S_aug, master_aug = self.HtrgGAT_layer_ST12(
            out_T1, out_S1, master=master1)
        out_T1 = out_T1 + out_T_aug
        out_S1 = out_S1 + out_S_aug
        master1 = master1 + master_aug

        # inference 2
        out_T2, out_S2, master2 = self.HtrgGAT_layer_ST21(
            out_T, out_S, master=self.master2)
        out_S2 = self.pool_hS2(out_S2)
        out_T2 = self.pool_hT2(out_T2)

        out_T_aug, out_S_aug, master_aug = self.HtrgGAT_layer_ST22(
            out_T2, out_S2, master=master2)
        out_T2 = out_T2 + out_T_aug
        out_S2 = out_S2 + out_S_aug
        master2 = master2 + master_aug

        out_T1 = self.drop_way(out_T1)
        out_T2 = self.drop_way(out_T2)
        out_S1 = self.drop_way(out_S1)
        out_S2 = self.drop_way(out_S2)
        master1 = self.drop_way(master1)
        master2 = self.drop_way(master2)

        out_T = torch.max(out_T1, out_T2)
        out_S = torch.max(out_S1, out_S2)
        master = torch.max(master1, master2)

        # Readout operation
        T_max, _ = torch.max(torch.abs(out_T), dim=1)
        T_avg = torch.mean(out_T, dim=1)

        S_max, _ = torch.max(torch.abs(out_S), dim=1)
        S_avg = torch.mean(out_S, dim=1)

        last_hidden = torch.cat(
            [T_max, T_avg, S_max, S_avg, master.squeeze(1)], dim=1)

        last_hidden = self.drop(last_hidden)
        output = self.out_layer(last_hidden)

        return output


class Distil_W2V2_AASISTL(nn.Module):
    def __init__(self, device):
        super().__init__()
        # AASIST parameters
        filts = [128, [1, 32], [32, 32], [32, 24], [24, 24]]
        gat_dims = [24, 32]
        pool_ratios = [0.4, 0.5, 0.7, 0.5]
        temperatures = [2.0, 2.0, 100.0, 100.0]

        ####
        # create network wav2vec 2.0
        ####
        self.ssl_model = DistilSSLModel(device)
        self.LL = nn.Linear(self.ssl_model.out_dim, 128)
        self.first_bn = nn.BatchNorm2d(num_features=1)
        self.first_bn1 = nn.BatchNorm2d(num_features=24)
        self.drop = nn.Dropout(0.5, inplace=True)
        self.drop_way = nn.Dropout(0.2, inplace=True)
        self.selu = nn.SELU(inplace=True)

        # RawNet2 encoder
        self.encoder = nn.Sequential(
            nn.Sequential(Residual_block(nb_filts=filts[1], first=True)),
            nn.Sequential(Residual_block(nb_filts=filts[2])),
            nn.Sequential(Residual_block(nb_filts=filts[3])),
            nn.Sequential(Residual_block(nb_filts=filts[4])),
            nn.Sequential(Residual_block(nb_filts=filts[4])),
            nn.Sequential(Residual_block(nb_filts=filts[4])))

        self.attention = nn.Sequential(
            nn.Conv2d(24, 128, kernel_size=(1, 1)),
            nn.SELU(inplace=True),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 24, kernel_size=(1, 1)),

        )
        # position encoding
        self.pos_S = nn.Parameter(torch.randn(1, 42, filts[-1][-1]))

        self.master1 = nn.Parameter(torch.randn(1, 1, gat_dims[0]))
        self.master2 = nn.Parameter(torch.randn(1, 1, gat_dims[0]))

        # Graph module
        self.GAT_layer_S = GraphAttentionLayer(filts[-1][-1],
                                               gat_dims[0],
                                               temperature=temperatures[0])
        self.GAT_layer_T = GraphAttentionLayer(filts[-1][-1],
                                               gat_dims[0],
                                               temperature=temperatures[1])
        # HS-GAL layer
        self.HtrgGAT_layer_ST11 = HtrgGraphAttentionLayer(
            gat_dims[0], gat_dims[1], temperature=temperatures[2])
        self.HtrgGAT_layer_ST12 = HtrgGraphAttentionLayer(
            gat_dims[1], gat_dims[1], temperature=temperatures[2])
        self.HtrgGAT_layer_ST21 = HtrgGraphAttentionLayer(
            gat_dims[0], gat_dims[1], temperature=temperatures[2])
        self.HtrgGAT_layer_ST22 = HtrgGraphAttentionLayer(
            gat_dims[1], gat_dims[1], temperature=temperatures[2])

        # Graph pooling layers
        self.pool_S = GraphPool(pool_ratios[0], gat_dims[0], 0.3)
        self.pool_T = GraphPool(pool_ratios[1], gat_dims[0], 0.3)
        self.pool_hS1 = GraphPool(pool_ratios[2], gat_dims[1], 0.3)
        self.pool_hT1 = GraphPool(pool_ratios[2], gat_dims[1], 0.3)

        self.pool_hS2 = GraphPool(pool_ratios[2], gat_dims[1], 0.3)
        self.pool_hT2 = GraphPool(pool_ratios[2], gat_dims[1], 0.3)

        self.out_layer = nn.Linear(5 * gat_dims[1], 2)

    def forward(self, x):
        # -------pre-trained Wav2vec model fine tunning ------------------------##
        x_ssl_feat = self.ssl_model(x.squeeze(-1))
        x = self.LL(x_ssl_feat)  # (bs,frame_number,feat_out_dim)

        # post-processing on front-end features
        x = x.transpose(1, 2)  # (bs,feat_out_dim,frame_number)
        x = x.unsqueeze(dim=1)  # add channel
        x = F.max_pool2d(x, (3, 3))
        x = self.first_bn(x)
        x = self.selu(x)

        # RawNet2-based encoder
        x = self.encoder(x)
        x = self.first_bn1(x)
        x = self.selu(x)

        w = self.attention(x)

        # ------------SA for spectral feature-------------#
        w1 = F.softmax(w, dim=-1)
        m = torch.sum(x * w1, dim=-1)
        e_S = m.transpose(1, 2) + self.pos_S

        # graph module layer
        gat_S = self.GAT_layer_S(e_S)
        out_S = self.pool_S(gat_S)  # (#bs, #node, #dim)

        # ------------SA for temporal feature-------------#
        w2 = F.softmax(w, dim=-2)
        m1 = torch.sum(x * w2, dim=-2)

        e_T = m1.transpose(1, 2)

        # graph module layer
        gat_T = self.GAT_layer_T(e_T)
        out_T = self.pool_T(gat_T)

        # learnable master node
        master1 = self.master1.expand(x.size(0), -1, -1)
        master2 = self.master2.expand(x.size(0), -1, -1)

        # inference 1
        out_T1, out_S1, master1 = self.HtrgGAT_layer_ST11(
            out_T, out_S, master=self.master1)

        out_S1 = self.pool_hS1(out_S1)
        out_T1 = self.pool_hT1(out_T1)

        out_T_aug, out_S_aug, master_aug = self.HtrgGAT_layer_ST12(
            out_T1, out_S1, master=master1)
        out_T1 = out_T1 + out_T_aug
        out_S1 = out_S1 + out_S_aug
        master1 = master1 + master_aug

        # inference 2
        out_T2, out_S2, master2 = self.HtrgGAT_layer_ST21(
            out_T, out_S, master=self.master2)
        out_S2 = self.pool_hS2(out_S2)
        out_T2 = self.pool_hT2(out_T2)

        out_T_aug, out_S_aug, master_aug = self.HtrgGAT_layer_ST22(
            out_T2, out_S2, master=master2)
        out_T2 = out_T2 + out_T_aug
        out_S2 = out_S2 + out_S_aug
        master2 = master2 + master_aug

        out_T1 = self.drop_way(out_T1)
        out_T2 = self.drop_way(out_T2)
        out_S1 = self.drop_way(out_S1)
        out_S2 = self.drop_way(out_S2)
        master1 = self.drop_way(master1)
        master2 = self.drop_way(master2)

        out_T = torch.max(out_T1, out_T2)
        out_S = torch.max(out_S1, out_S2)
        master = torch.max(master1, master2)

        # Readout operation
        T_max, _ = torch.max(torch.abs(out_T), dim=1)
        T_avg = torch.mean(out_T, dim=1)

        S_max, _ = torch.max(torch.abs(out_S), dim=1)
        S_avg = torch.mean(out_S, dim=1)

        last_hidden = torch.cat(
            [T_max, T_avg, S_max, S_avg, master.squeeze(1)], dim=1)

        last_hidden = self.drop(last_hidden)
        output = self.out_layer(last_hidden)

        return output


@register_model(key='SelfDistil_W2V2BASE_AASISTL')
class SelfDistil_W2V2BASE_AASISTL(nn.Module):
    def __init__(self, device, ssl_cpkt_path):
        super().__init__()
        # AASIST parameters
        filts = [128, [1, 32], [32, 32], [32, 24], [24, 24]]
        gat_dims = [24, 32]
        pool_ratios = [0.4, 0.5, 0.7, 0.5]
        temperatures = [2.0, 2.0, 100.0, 100.0]

        ####
        # create network wav2vec 2.0
        ####
        self.ssl_model = SSLModel(device, ssl_cpkt_path, 768).to(device)
        self.LL = nn.Linear(self.ssl_model.out_dim, 128)
        self.first_bn = nn.BatchNorm2d(num_features=1)
        self.first_bn1 = nn.BatchNorm2d(num_features=24)
        self.drop = nn.Dropout(0.5, inplace=True)
        self.drop_way = nn.Dropout(0.2, inplace=True)
        self.selu = nn.SELU(inplace=True)

        # RawNet2 encoder
        self.encoder = nn.Sequential(
            nn.Sequential(Residual_block(nb_filts=filts[1], first=True)),
            nn.Sequential(Residual_block(nb_filts=filts[2])),
            nn.Sequential(Residual_block(nb_filts=filts[3])),
            nn.Sequential(Residual_block(nb_filts=filts[4])),
            nn.Sequential(Residual_block(nb_filts=filts[4])),
            nn.Sequential(Residual_block(nb_filts=filts[4])))

        self.attention = nn.Sequential(
            nn.Conv2d(24, 128, kernel_size=(1, 1)),
            nn.SELU(inplace=True),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 24, kernel_size=(1, 1)),

        )
        # position encoding
        self.pos_S = nn.Parameter(torch.randn(1, 42, filts[-1][-1]))

        self.master1 = nn.Parameter(torch.randn(1, 1, gat_dims[0]))
        self.master2 = nn.Parameter(torch.randn(1, 1, gat_dims[0]))

        # Graph module
        self.GAT_layer_S = GraphAttentionLayer(filts[-1][-1],
                                               gat_dims[0],
                                               temperature=temperatures[0])
        self.GAT_layer_T = GraphAttentionLayer(filts[-1][-1],
                                               gat_dims[0],
                                               temperature=temperatures[1])
        # HS-GAL layer
        self.HtrgGAT_layer_ST11 = HtrgGraphAttentionLayer(
            gat_dims[0], gat_dims[1], temperature=temperatures[2])
        self.HtrgGAT_layer_ST12 = HtrgGraphAttentionLayer(
            gat_dims[1], gat_dims[1], temperature=temperatures[2])
        self.HtrgGAT_layer_ST21 = HtrgGraphAttentionLayer(
            gat_dims[0], gat_dims[1], temperature=temperatures[2])
        self.HtrgGAT_layer_ST22 = HtrgGraphAttentionLayer(
            gat_dims[1], gat_dims[1], temperature=temperatures[2])

        # Graph pooling layers
        self.pool_S = GraphPool(pool_ratios[0], gat_dims[0], 0.3)
        self.pool_T = GraphPool(pool_ratios[1], gat_dims[0], 0.3)
        self.pool_hS1 = GraphPool(pool_ratios[2], gat_dims[1], 0.3)
        self.pool_hT1 = GraphPool(pool_ratios[2], gat_dims[1], 0.3)

        self.pool_hS2 = GraphPool(pool_ratios[2], gat_dims[1], 0.3)
        self.pool_hT2 = GraphPool(pool_ratios[2], gat_dims[1], 0.3)

        self.middle_fc1 = nn.Linear(1008, 2)
        self.middle_fc2 = nn.Linear(1608, 2)
        self.middle_fc3 = nn.Linear(352, 2)
        self.middle_fc4 = nn.Linear(736, 2)
        self.middle_fc5 = nn.Linear(352, 2)
        self.middle_fc6 = nn.Linear(736, 2)

        self.out_layer = nn.Linear(5 * gat_dims[1], 2)

    def forward(self, x):
        # -------pre-trained Wav2vec model fine tunning ------------------------##
        x_ssl_feat = self.ssl_model(x.squeeze(-1))
        x = self.LL(x_ssl_feat)  # (bs,frame_number,feat_out_dim)

        x_ssl_feat = x

        # post-processing on front-end features
        x = x.transpose(1, 2)  # (bs,feat_out_dim,frame_number)
        x = x.unsqueeze(dim=1)  # add channel
        x = F.max_pool2d(x, (3, 3))
        x = self.first_bn(x)
        x = self.selu(x)

        # RawNet2-based encoder
        x = self.encoder(x)
        x = self.first_bn1(x)
        x = self.selu(x)

        w = self.attention(x)

        # ------------SA for spectral feature-------------#
        w1 = F.softmax(w, dim=-1)
        m = torch.sum(x * w1, dim=-1)
        e_S = m.transpose(1, 2) + self.pos_S

        # Extract spectral feature for middle FC 1

        spectral_output = self.middle_fc1(e_S.reshape(
            e_S.size(0), -1))  # Flatten and apply Linear

        # graph module layer
        gat_S = self.GAT_layer_S(e_S)
        out_S = self.pool_S(gat_S)  # (#bs, #node, #dim)

        # ------------SA for temporal feature-------------#
        w2 = F.softmax(w, dim=-2)
        m1 = torch.sum(x * w2, dim=-2)

        e_T = m1.transpose(1, 2)
        # Extract temporal feature

        temporal_output = self.middle_fc2(e_T.reshape(
            e_T.size(0), -1))  # Flatten and apply Linear

        # graph module layer
        gat_T = self.GAT_layer_T(e_T)
        out_T = self.pool_T(gat_T)

        # learnable master node
        master1 = self.master1.expand(x.size(0), -1, -1)
        master2 = self.master2.expand(x.size(0), -1, -1)

        # inference 1
        out_T1, out_S1, master1 = self.HtrgGAT_layer_ST11(
            out_T, out_S, master=self.master1)

        out_S1 = self.pool_hS1(out_S1)
        out_T1 = self.pool_hT1(out_T1)

        # Extract output from Graph modules

        # e_S_flattened = e_S.view(e_S.size(0), -1)  # Reshapes to [1, 42*64]
        # e_T_flattened = e_T.view(e_T.size(0), -1)  # Reshapes to [1, 16*64]

        graph_output_S = self.middle_fc3(out_S1.reshape(
            out_S1.size(0), -1))  # Flatten and apply Linear
        middle_feature1 = out_S1.reshape(out_S1.size(0), -1)
        graph_output_T = self.middle_fc4(out_T1.reshape(
            out_T1.size(0), -1))  # Flatten and apply Linear
        middle_feature2 = out_T1.reshape(out_T1.size(0), -1)

        out_T_aug, out_S_aug, master_aug = self.HtrgGAT_layer_ST12(
            out_T1, out_S1, master=master1)
        out_T1 = out_T1 + out_T_aug
        out_S1 = out_S1 + out_S_aug
        master1 = master1 + master_aug

        # inference 2
        out_T2, out_S2, master2 = self.HtrgGAT_layer_ST21(
            out_T, out_S, master=self.master2)
        out_S2 = self.pool_hS2(out_S2)
        out_T2 = self.pool_hT2(out_T2)

        out_T_aug, out_S_aug, master_aug = self.HtrgGAT_layer_ST22(
            out_T2, out_S2, master=master2)
        out_T2 = out_T2 + out_T_aug
        out_S2 = out_S2 + out_S_aug
        master2 = master2 + master_aug

        out_T1 = self.drop_way(out_T1)
        out_T2 = self.drop_way(out_T2)
        out_S1 = self.drop_way(out_S1)
        out_S2 = self.drop_way(out_S2)

        hs_gal_output_S = self.middle_fc5(out_S2.reshape(
            out_S2.size(0), -1))  # Flatten and apply Linear
        final_feature1 = out_S2.reshape(out_S2.size(0), -1)
        hs_gal_output_T = self.middle_fc6(out_T2.reshape(
            out_T2.size(0), -1))  # Flatten and apply Linear
        final_feature2 = out_T2.reshape(out_T2.size(0), -1)

        master1 = self.drop_way(master1)
        master2 = self.drop_way(master2)

        out_T = torch.max(out_T1, out_T2)
        out_S = torch.max(out_S1, out_S2)
        master = torch.max(master1, master2)

        # Readout operation
        T_max, _ = torch.max(torch.abs(out_T), dim=1)
        T_avg = torch.mean(out_T, dim=1)

        S_max, _ = torch.max(torch.abs(out_S), dim=1)
        S_avg = torch.mean(out_S, dim=1)

        last_hidden = torch.cat(
            [T_max, T_avg, S_max, S_avg, master.squeeze(1)], dim=1)

        last_hidden = self.drop(last_hidden)
        output = self.out_layer(last_hidden)

        return output, spectral_output, temporal_output, graph_output_S, graph_output_T, hs_gal_output_S, hs_gal_output_T, middle_feature1, middle_feature2, final_feature1, final_feature2, x_ssl_feat


@register_model(key='Distil_W2V2BASE_AASISTL')
class Distil_W2V2BASE_AASISTL(nn.Module):
    def __init__(self, device, ssl_cpkt_path):
        super().__init__()
        # AASIST parameters
        filts = [128, [1, 32], [32, 32], [32, 24], [24, 24]]
        gat_dims = [24, 32]
        pool_ratios = [0.4, 0.5, 0.7, 0.5]
        temperatures = [2.0, 2.0, 100.0, 100.0]

        ####
        # create network wav2vec 2.0
        ####
        self.ssl_model = SSLModel(device, ssl_cpkt_path, 768).to(device)
        self.LL = nn.Linear(self.ssl_model.out_dim, 128)
        self.first_bn = nn.BatchNorm2d(num_features=1)
        self.first_bn1 = nn.BatchNorm2d(num_features=24)
        self.drop = nn.Dropout(0.5, inplace=True)
        self.drop_way = nn.Dropout(0.2, inplace=True)
        self.selu = nn.SELU(inplace=True)

        # RawNet2 encoder
        self.encoder = nn.Sequential(
            nn.Sequential(Residual_block(nb_filts=filts[1], first=True)),
            nn.Sequential(Residual_block(nb_filts=filts[2])),
            nn.Sequential(Residual_block(nb_filts=filts[3])),
            nn.Sequential(Residual_block(nb_filts=filts[4])),
            nn.Sequential(Residual_block(nb_filts=filts[4])),
            nn.Sequential(Residual_block(nb_filts=filts[4])))

        self.attention = nn.Sequential(
            nn.Conv2d(24, 128, kernel_size=(1, 1)),
            nn.SELU(inplace=True),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 24, kernel_size=(1, 1)),

        )
        # position encoding
        self.pos_S = nn.Parameter(torch.randn(1, 42, filts[-1][-1]))

        self.master1 = nn.Parameter(torch.randn(1, 1, gat_dims[0]))
        self.master2 = nn.Parameter(torch.randn(1, 1, gat_dims[0]))

        # Graph module
        self.GAT_layer_S = GraphAttentionLayer(filts[-1][-1],
                                               gat_dims[0],
                                               temperature=temperatures[0])
        self.GAT_layer_T = GraphAttentionLayer(filts[-1][-1],
                                               gat_dims[0],
                                               temperature=temperatures[1])
        # HS-GAL layer
        self.HtrgGAT_layer_ST11 = HtrgGraphAttentionLayer(
            gat_dims[0], gat_dims[1], temperature=temperatures[2])
        self.HtrgGAT_layer_ST12 = HtrgGraphAttentionLayer(
            gat_dims[1], gat_dims[1], temperature=temperatures[2])
        self.HtrgGAT_layer_ST21 = HtrgGraphAttentionLayer(
            gat_dims[0], gat_dims[1], temperature=temperatures[2])
        self.HtrgGAT_layer_ST22 = HtrgGraphAttentionLayer(
            gat_dims[1], gat_dims[1], temperature=temperatures[2])

        # Graph pooling layers
        self.pool_S = GraphPool(pool_ratios[0], gat_dims[0], 0.3)
        self.pool_T = GraphPool(pool_ratios[1], gat_dims[0], 0.3)
        self.pool_hS1 = GraphPool(pool_ratios[2], gat_dims[1], 0.3)
        self.pool_hT1 = GraphPool(pool_ratios[2], gat_dims[1], 0.3)

        self.pool_hS2 = GraphPool(pool_ratios[2], gat_dims[1], 0.3)
        self.pool_hT2 = GraphPool(pool_ratios[2], gat_dims[1], 0.3)

        self.out_layer = nn.Linear(5 * gat_dims[1], 2)

    def forward(self, x):
        # -------pre-trained Wav2vec model fine tunning ------------------------##
        x_ssl_feat = self.ssl_model(x.squeeze(-1))
        x = self.LL(x_ssl_feat)  # (bs,frame_number,feat_out_dim)

        # post-processing on front-end features
        x = x.transpose(1, 2)  # (bs,feat_out_dim,frame_number)
        x = x.unsqueeze(dim=1)  # add channel
        x = F.max_pool2d(x, (3, 3))
        x = self.first_bn(x)
        x = self.selu(x)

        # RawNet2-based encoder
        x = self.encoder(x)
        x = self.first_bn1(x)
        x = self.selu(x)

        w = self.attention(x)

        # ------------SA for spectral feature-------------#
        w1 = F.softmax(w, dim=-1)
        m = torch.sum(x * w1, dim=-1)
        e_S = m.transpose(1, 2) + self.pos_S

        # graph module layer
        gat_S = self.GAT_layer_S(e_S)
        out_S = self.pool_S(gat_S)  # (#bs, #node, #dim)

        # ------------SA for temporal feature-------------#
        w2 = F.softmax(w, dim=-2)
        m1 = torch.sum(x * w2, dim=-2)

        e_T = m1.transpose(1, 2)

        # graph module layer
        gat_T = self.GAT_layer_T(e_T)
        out_T = self.pool_T(gat_T)

        # learnable master node
        master1 = self.master1.expand(x.size(0), -1, -1)
        master2 = self.master2.expand(x.size(0), -1, -1)

        # inference 1
        out_T1, out_S1, master1 = self.HtrgGAT_layer_ST11(
            out_T, out_S, master=self.master1)

        out_S1 = self.pool_hS1(out_S1)
        out_T1 = self.pool_hT1(out_T1)

        out_T_aug, out_S_aug, master_aug = self.HtrgGAT_layer_ST12(
            out_T1, out_S1, master=master1)
        out_T1 = out_T1 + out_T_aug
        out_S1 = out_S1 + out_S_aug
        master1 = master1 + master_aug

        # inference 2
        out_T2, out_S2, master2 = self.HtrgGAT_layer_ST21(
            out_T, out_S, master=self.master2)
        out_S2 = self.pool_hS2(out_S2)
        out_T2 = self.pool_hT2(out_T2)

        out_T_aug, out_S_aug, master_aug = self.HtrgGAT_layer_ST22(
            out_T2, out_S2, master=master2)
        out_T2 = out_T2 + out_T_aug
        out_S2 = out_S2 + out_S_aug
        master2 = master2 + master_aug

        out_T1 = self.drop_way(out_T1)
        out_T2 = self.drop_way(out_T2)
        out_S1 = self.drop_way(out_S1)
        out_S2 = self.drop_way(out_S2)
        master1 = self.drop_way(master1)
        master2 = self.drop_way(master2)

        out_T = torch.max(out_T1, out_T2)
        out_S = torch.max(out_S1, out_S2)
        master = torch.max(master1, master2)

        # Readout operation
        T_max, _ = torch.max(torch.abs(out_T), dim=1)
        T_avg = torch.mean(out_T, dim=1)

        S_max, _ = torch.max(torch.abs(out_S), dim=1)
        S_avg = torch.mean(out_S, dim=1)

        last_hidden = torch.cat(
            [T_max, T_avg, S_max, S_avg, master.squeeze(1)], dim=1)

        last_hidden = self.drop(last_hidden)
        output = self.out_layer(last_hidden)

        return output


@register_model(key='Distil_W2V2BASE_Linear')
class Distil_W2V2BASE_Linear(nn.Module):
    def __init__(self, device, ssl_cpkt_path):
        super().__init__()
        ####
        # create network wav2vec 2.0
        ####
        self.ssl_model = SSLModel(device, ssl_cpkt_path, 768).to(device)
        self.LL = nn.Linear(self.ssl_model.out_dim, 128)
        self.backend = BackEnd(128, 128, 2, 0.5, False)

        self.relu = nn.ReLU()

    def forward(self, x):
        # -------pre-trained Wav2vec model fine tunning ------------------------##
        x_ssl_feat = self.ssl_model(x.squeeze(-1))
        x = self.LL(x_ssl_feat)  # (bs,frame_number,feat_out_dim)
        x = self.relu(x)
        output = self.backend(x)
        return output


@register_model(key='Distil_XLSR_N_Trans_Layer_Linear')
class Distil_XLSR_N_Trans_Layer_Linear(nn.Module):
    def __init__(self, device, ssl_cpkt_path=None, **kwargs):
        super().__init__()
        ####
        # create network wav2vec 2.0
        ####
        self.ssl_model = My_XLSR_FE(device, **kwargs).to(device)
        self.LL = nn.Linear(self.ssl_model.out_dim, 128)
        self.backend = BackEnd(128, 128, 2, 0.5, False)

        self.relu = nn.ReLU()

    def forward(self, x):
        # -------pre-trained Wav2vec model fine tunning ------------------------##
        x_ssl_feat = self.ssl_model(x.squeeze(-1))
        x = self.LL(x_ssl_feat)  # (bs,frame_number,feat_out_dim)
        x = self.relu(x)
        output = self.backend(x)
        return output


@register_model(key='Distil_W2V2BASE_VIB')
class Distil_W2V2BASE_VIB(nn.Module):
    def __init__(self, device, ssl_cpkt_path):
        super().__init__()
        ####
        # create network wav2vec 2.0
        ####
        self.ssl_model = SSLModel(device, ssl_cpkt_path, 768).to(device)
        self.LL = nn.Linear(self.ssl_model.out_dim, 128)
        self.VIB = VIB(128, 128, 64)
        self.backend = BackEndVIB(64, 64, 2, 0.5, False)

        self.gelu = nn.GELU()

    def forward(self, x):
        # -------pre-trained Wav2vec model fine tunning ------------------------##
        x_ssl_feat = self.ssl_model(x.squeeze(-1))
        x = self.LL(x_ssl_feat)  # (bs,frame_number,feat_out_dim)
        x = self.gelu(x)
        x, decoded, mu, logvar = self.VIB(x)

        output = self.backend(x)
        return output


@register_model(key='Distil_XLSR_N_Trans_Layer_VIB')
class Distil_XLSR_N_Trans_Layer_VIB(nn.Module):
    def __init__(self, device, ssl_cpkt_path=None, **kwargs):
        super().__init__()
        ####
        # create network wav2vec 2.0
        ####
        self.ssl_model = My_XLSR_FE(device, **kwargs).to(device)
        # self.ssl_model = SSLModel(device, ssl_cpkt_path, 1024).to(device)
        self.LL = nn.Linear(self.ssl_model.out_dim, 128)
        self.VIB = VIB(128, 128, 64)
        self.backend = BackEndVIB(64, 64, 2, 0.5, False)
        self.gelu = nn.GELU()

    def forward(self, x):
        # -------pre-trained Wav2vec model fine tunning ------------------------##
        x_ssl_feat = self.ssl_model.extract_feat(x.squeeze(-1))
        x = self.LL(x_ssl_feat)  # (bs,frame_number,feat_out_dim)
        x = self.gelu(x)
        x, decoded, mu, logvar = self.VIB(x)
        output = self.backend(x)
        return output


@register_model(key='Self_Distil_XLSR_N_Trans_Layer_VIB')
class Self_Distil_XLSR_N_Trans_Layer_VIB(nn.Module):
    def __init__(self, device, ssl_cpkt_path=None, **kwargs):
        super().__init__()
        ####
        # create network wav2vec 2.0
        ####
        self.ssl_model = My_XLSR_FE(device, **kwargs).to(device)
        # self.ssl_model = SSLModel(device, ssl_cpkt_path, 1024).to(device)

        # Handler for last transformer block
        self.LL = nn.Linear(self.ssl_model.out_dim, 128)
        self.VIB = VIB(128, 128, 64)
        self.backend = BackEndVIB(64, 64, 2, 0.5, False)
        self.gelu = nn.GELU()

        # Handler for other transformer blocks

        # Transformer block 1
        self.LL1 = nn.Linear(self.ssl_model.out_dim, 128)
        self.VIB1 = VIB(128, 128, 64)
        self.backend1 = BackEndVIB(64, 64, 2, 0.5, False)

        # Transformer block 2
        self.LL2 = nn.Linear(self.ssl_model.out_dim, 128)
        self.VIB2 = VIB(128, 128, 64)
        self.backend2 = BackEndVIB(64, 64, 2, 0.5, False)

        # Transformer block 3
        self.LL3 = nn.Linear(self.ssl_model.out_dim, 128)
        self.VIB3 = VIB(128, 128, 64)
        self.backend3 = BackEndVIB(64, 64, 2, 0.5, False)

        # Transformer block 4
        self.LL4 = nn.Linear(self.ssl_model.out_dim, 128)
        self.VIB4 = VIB(128, 128, 64)
        self.backend4 = BackEndVIB(64, 64, 2, 0.5, False)

    def forward(self, x):
        # -------pre-trained Wav2vec model fine tunning ------------------------##

        # extract features from multiple transformer blocks
        # Returns a list of features from multiple transformer blocks list((bs, frame_number, feat_out_dim))
        mul_features = self.ssl_model.extract_layer_results(x.squeeze(-1))
        logits = []
        x_ssl_feats = []

        for i, x_ssl_feat in enumerate(mul_features):
            # Extracting the first element from the tuple
            x_ssl_feat = x_ssl_feat[0].transpose(0, 1)
            x_ssl_feats.append(x_ssl_feat)
            if i == 0:
                x = self.LL1(x_ssl_feat)
                x = self.gelu(x)
                x, decoded, mu, logvar = self.VIB1(x)
                output = self.backend1(x)
                logits.append(output)
            elif i == 1:
                x = self.LL2(x_ssl_feat)
                x = self.gelu(x)
                x, decoded, mu, logvar = self.VIB2(x)
                output = self.backend2(x)
                logits.append(output)
            elif i == 2:
                x = self.LL3(x_ssl_feat)
                x = self.gelu(x)
                x, decoded, mu, logvar = self.VIB3(x)
                output = self.backend3(x)
                logits.append(output)
            elif i == 3:
                x = self.LL4(x_ssl_feat)
                x = self.gelu(x)
                x, decoded, mu, logvar = self.VIB4(x)
                output = self.backend4(x)
                logits.append(output)
            elif i == 4:
                x = self.LL(x_ssl_feat)
                x = self.gelu(x)
                x, decoded, mu, logvar = self.VIB(x)
                output = self.backend(x)
                logits.append(output)

        return logits, x_ssl_feats


@register_model(key='Distil_Wav2vec2_Custom_VIB')
class Distil_Wav2vec2_Custom_VIB(nn.Module):

    def __init__(self, device, ssl_cpkt_path=None, **kwargs):
        super().__init__()
        ####
        # create network wav2vec 2.0
        ####
        self.ssl_model = Custom_Wav2Vec2_Fe(device, **kwargs).to(device)
        self.LL = nn.Linear(self.ssl_model.out_dim, 128)
        self.VIB = VIB(128, 128, 64)
        self.backend = BackEndVIB(64, 64, 2, 0.5, False)
        self.gelu = nn.GELU()

    def forward(self, x):
        # -------pre-trained Wav2vec model fine tunning ------------------------##
        x_ssl_feat = self.ssl_model(x.squeeze(-1))
        x = self.LL(x_ssl_feat)  # (bs,frame_number,feat_out_dim)
        x = self.gelu(x)
        x, decoded, mu, logvar = self.VIB(x)
        output = self.backend(x)
        return output


@register_model(key='Distil_W2V2BASE_ConvNeXt_COAASISTL')
class Distil_W2V2BASE_ConvNeXt_COAASISTL(nn.Module):
    def __init__(self, device, ssl_cpkt_path, emb_size=128, heads=4, ffmult=4, exp_fac=2, kernel_size=31, n_encoders=4):
        super().__init__()
        self.dim_head = int(emb_size/heads)
        # AASIST parameters
        filts = [128, [1, 32], [32, 32], [32, 24], [24, 24]]
        gat_dims = [24, 32]
        pool_ratios = [0.4, 0.5, 0.7, 0.5]
        temperatures = [2.0, 2.0, 100.0, 100.0]

        ####
        # create network wav2vec 2.0
        ####
        self.ssl_model = SSLModel(device, ssl_cpkt_path, 768).to(device)
        self.LL = nn.Linear(self.ssl_model.out_dim, 128)
        self.first_bn = nn.BatchNorm2d(num_features=1)
        self.first_bn1 = nn.BatchNorm2d(num_features=24)
        self.drop = nn.Dropout(0.5, inplace=True)
        self.drop_way = nn.Dropout(0.2, inplace=True)
        self.selu = nn.SELU(inplace=True)

        # RawNet2 encoder
        self.encoder = nn.Sequential(
            nn.Sequential(Residual_block(nb_filts=filts[1], first=True)),
            nn.Sequential(Residual_block(nb_filts=filts[2])),
            nn.Sequential(Residual_block(nb_filts=filts[3])),
            nn.Sequential(Residual_block(nb_filts=filts[4])),
            nn.Sequential(Residual_block(nb_filts=filts[4])),
            nn.Sequential(Residual_block(nb_filts=filts[4])))

        # ConvNeXt encoder
        self.convnext_encoder = ConvNeXt(dims=[24, 24, 24, 24])

        # Conformer encoder
        self.encoder_conformer = _get_clones(ConformerBlock(dim=emb_size, dim_head=self.dim_head, heads=heads,
                                                            ff_mult=ffmult, conv_expansion_factor=exp_fac, conv_kernel_size=kernel_size),
                                             n_encoders)

        self.attention = nn.Sequential(
            nn.Conv2d(24, 128, kernel_size=(1, 1)),
            nn.SELU(inplace=True),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 24, kernel_size=(1, 1)),

        )
        # position encoding
        self.pos_S = nn.Parameter(torch.randn(1, 42, filts[-1][-1]))

        self.master1 = nn.Parameter(torch.randn(1, 1, gat_dims[0]))
        self.master2 = nn.Parameter(torch.randn(1, 1, gat_dims[0]))

        # Graph module
        self.GAT_layer_S = GraphAttentionLayer(filts[-1][-1],
                                               gat_dims[0],
                                               temperature=temperatures[0])
        self.GAT_layer_T = GraphAttentionLayer(filts[-1][-1],
                                               gat_dims[0],
                                               temperature=temperatures[1])
        # HS-GAL layer
        self.HtrgGAT_layer_ST11 = HtrgGraphAttentionLayer(
            gat_dims[0], gat_dims[1], temperature=temperatures[2])
        self.HtrgGAT_layer_ST12 = HtrgGraphAttentionLayer(
            gat_dims[1], gat_dims[1], temperature=temperatures[2])
        self.HtrgGAT_layer_ST21 = HtrgGraphAttentionLayer(
            gat_dims[0], gat_dims[1], temperature=temperatures[2])
        self.HtrgGAT_layer_ST22 = HtrgGraphAttentionLayer(
            gat_dims[1], gat_dims[1], temperature=temperatures[2])

        # Graph pooling layers
        self.pool_S = GraphPool(pool_ratios[0], gat_dims[0], 0.3)
        self.pool_T = GraphPool(pool_ratios[1], gat_dims[0], 0.3)
        self.pool_hS1 = GraphPool(pool_ratios[2], gat_dims[1], 0.3)
        self.pool_hT1 = GraphPool(pool_ratios[2], gat_dims[1], 0.3)

        self.pool_hS2 = GraphPool(pool_ratios[2], gat_dims[1], 0.3)
        self.pool_hT2 = GraphPool(pool_ratios[2], gat_dims[1], 0.3)

        self.out_layer = nn.Linear(5 * gat_dims[1], 2)

    def forward(self, x):
        # -------pre-trained Wav2vec model fine tunning ------------------------##
        x_ssl_feat = self.ssl_model(x.squeeze(-1))
        x = self.LL(x_ssl_feat)  # (bs,frame_number,feat_out_dim)

        # post-processing on front-end features
        # x = x.transpose(1, 2)  # (bs,feat_out_dim,frame_number)
        x = x.unsqueeze(dim=1)  # add channel
        # x = F.max_pool2d(x, (3, 3))
        x = self.first_bn(x)
        x = self.selu(x)

        x = x.squeeze(dim=1)
        for layer in self.encoder_conformer:
            x = layer(x)

        x = x.transpose(1, 2)  # (bs,feat_out_dim,frame_number)
        x = x.unsqueeze(dim=1)  # add channel
        x = F.max_pool2d(x, (3, 3))

        # RawNet2-based encoder
        x = self.encoder(x)
        x = self.convnext_encoder(x)

        x = self.first_bn1(x)
        x = self.selu(x)

        w = self.attention(x)

        # ------------SA for spectral feature-------------#
        w1 = F.softmax(w, dim=-1)
        m = torch.sum(x * w1, dim=-1)
        e_S = m.transpose(1, 2) + self.pos_S

        # graph module layer
        gat_S = self.GAT_layer_S(e_S)
        out_S = self.pool_S(gat_S)  # (#bs, #node, #dim)

        # ------------SA for temporal feature-------------#
        w2 = F.softmax(w, dim=-2)
        m1 = torch.sum(x * w2, dim=-2)

        e_T = m1.transpose(1, 2)

        # graph module layer
        gat_T = self.GAT_layer_T(e_T)
        out_T = self.pool_T(gat_T)

        # learnable master node
        master1 = self.master1.expand(x.size(0), -1, -1)
        master2 = self.master2.expand(x.size(0), -1, -1)

        # inference 1
        out_T1, out_S1, master1 = self.HtrgGAT_layer_ST11(
            out_T, out_S, master=self.master1)

        out_S1 = self.pool_hS1(out_S1)
        out_T1 = self.pool_hT1(out_T1)

        out_T_aug, out_S_aug, master_aug = self.HtrgGAT_layer_ST12(
            out_T1, out_S1, master=master1)
        out_T1 = out_T1 + out_T_aug
        out_S1 = out_S1 + out_S_aug
        master1 = master1 + master_aug

        # inference 2
        out_T2, out_S2, master2 = self.HtrgGAT_layer_ST21(
            out_T, out_S, master=self.master2)
        out_S2 = self.pool_hS2(out_S2)
        out_T2 = self.pool_hT2(out_T2)

        out_T_aug, out_S_aug, master_aug = self.HtrgGAT_layer_ST22(
            out_T2, out_S2, master=master2)
        out_T2 = out_T2 + out_T_aug
        out_S2 = out_S2 + out_S_aug
        master2 = master2 + master_aug

        out_T1 = self.drop_way(out_T1)
        out_T2 = self.drop_way(out_T2)
        out_S1 = self.drop_way(out_S1)
        out_S2 = self.drop_way(out_S2)
        master1 = self.drop_way(master1)
        master2 = self.drop_way(master2)

        out_T = torch.max(out_T1, out_T2)
        out_S = torch.max(out_S1, out_S2)
        master = torch.max(master1, master2)

        # Readout operation
        T_max, _ = torch.max(torch.abs(out_T), dim=1)
        T_avg = torch.mean(out_T, dim=1)

        S_max, _ = torch.max(torch.abs(out_S), dim=1)
        S_avg = torch.mean(out_S, dim=1)

        last_hidden = torch.cat(
            [T_max, T_avg, S_max, S_avg, master.squeeze(1)], dim=1)

        last_hidden = self.drop(last_hidden)
        output = self.out_layer(last_hidden)

        return output


if __name__ == "__main__":
    # from torchinfo import summary
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # # model = W2V2_COAASIST(
    # #     device=device, ssl_cpkt_path='/datab/hungdx/Rawformer-implementation-anti-spoofing/xlsr2_300m.pt').to(device)
    # model = Distil_W2V2BASE_ConvNeXt_COAASISTL(
    #     device=device, ssl_cpkt_path='/datab/hungdx/KDW2V-AASISTL/wav2vec_small.pt').to(device)

    # summary(model, (1, 64600))
    model = Distil_XLSR_N_Trans_Layer_Linear(device=device)
    print(model)
    print("Done")
