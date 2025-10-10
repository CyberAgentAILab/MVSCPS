import torch
import torch.nn as nn
import torch.nn.functional as F

from core.registry import REG
from models.network_utils import get_encoding, get_mlp


@REG.register('model', name='neural_brdf')
class VolumeRadianceNear(nn.Module):
    def __init__(self, config):
        super(VolumeRadianceNear, self).__init__()
        self.config = config
        self.use_ae = self.config.get('use_ae', True)
        self.num_ae = self.config.get('num_ae', 5)

        self.n_output_dims = 3
        self.dir_encoding = get_encoding(3, self.config.dir_encoding_config)
        if self.use_ae:
            self.n_input_dims = int(self.config.input_feature_dim + self.num_ae)
        else:
            self.n_input_dims = int(self.config.input_feature_dim + self.dir_encoding.n_output_dims * 3)
        network = get_mlp(self.n_input_dims, self.n_output_dims, self.config.mlp_network_config)
        self.network = network

    def forward(self, spatial_features, N, V, L):
        N_embd = self.dir_encoding(N.view(-1, 3))
        V_embd = self.dir_encoding(V.view(-1, 3))
        L_embd = self.dir_encoding(L.view(-1, 3))
        vec_input = torch.cat([N_embd, V_embd, L_embd], dim=-1)

        NoV = torch.sum(N * V, dim=-1, keepdim=True)
        NoL = torch.sum(N * L, dim=-1, keepdim=True)
        H = F.normalize(V + L, p=2, dim=-1)
        NoH = torch.sum(N * H, dim=-1, keepdim=True)  # aka, cos(theta_h) in rusinkiewicz coordinates
        LoH = torch.sum(L * H, dim=-1, keepdim=True)  # aka, cos(theta_d) in rusinkiewicz coordinates

        if self.num_ae == 5:
            brdf_enc = [NoH, NoL, LoH, NoV, NoH**10]
        elif self.num_ae == 4:
            brdf_enc = [NoH, NoL, LoH, NoV]
        elif self.num_ae == 3:
            brdf_enc = [NoH, NoL, LoH]
        elif self.num_ae == 8:
            brdf_enc = [NoH, NoL, LoH, NoV, NoH ** 10, NoH ** 100, NoH ** 1000, NoH ** 10000]

        brdf_enc = torch.cat(brdf_enc, dim=-1)

        if self.use_ae:
            network_inp = torch.cat([spatial_features.view(-1, spatial_features.shape[-1]), brdf_enc], dim=-1)
        else:
            network_inp = torch.cat([spatial_features.view(-1, spatial_features.shape[-1]), vec_input], dim=-1)

        brdf_values = self.network(network_inp).view(*spatial_features.shape[:-1], self.n_output_dims).float()
        brdf_values = brdf_values / 1000

        return brdf_values

    def update_step(self, epoch, global_step):
        pass

    def regularizations(self, out):
        return {}

@REG.register('model', name='shadow-mapping')
class ShadowNet(nn.Module):
    def __init__(self, config):
        super(ShadowNet, self).__init__()
        self.config = config
        self.n_output_dims = 1

        self.dir_encoding = get_encoding(3, self.config.dir_encoding_config)
        self.n_input_dims = int(self.config.input_feature_dim + self.dir_encoding.n_output_dims + 1)
        network = get_mlp(self.n_input_dims, self.n_output_dims, self.config.mlp_network_config)
        self.network = network

    def forward(self, surface_features, shadow_value, V):
        view_dir_embd = self.dir_encoding(V.view(-1, 3))

        network_inp = [surface_features.view(-1, surface_features.shape[-1]), view_dir_embd, shadow_value.view(-1, 1)]
        network_inp = torch.cat(network_inp, dim=-1)
        shadow_refined = self.network(network_inp).view(*surface_features.shape[:-1], self.n_output_dims).float()
        return shadow_refined

    def update_step(self, epoch, global_step):
        pass

    def regularizations(self, out):
        return {}

