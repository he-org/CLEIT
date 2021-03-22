import torch
import torch.nn as nn
import math
from torch.nn import functional as F
from base_ae import BaseAE
from types_ import *
from typing import List


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class GaussianNoise(nn.Module):
    """Gaussian noise regularizer.

    Args:
        sigma (float, optional): relative standard deviation used to generate the
            noise. Relative means that it will be multiplied by the magnitude of
            the value your are adding the noise to. This means that sigma can be
            the same regardless of the scale of the vector.
        is_relative_detach (bool, optional): whether to detach the variable before
            computing the scale of the noise. If `False` then the scale of the noise
            won't be seen as a constant but something to optimize: this will bias the
            network to generate vectors with smaller values.
    """

    def __init__(self, sigma=0.1, is_relative_detach=True, device='cpu'):
        super().__init__()
        self.sigma = sigma
        self.is_relative_detach = is_relative_detach
        self.noise = torch.tensor(0).to(device)

    def forward(self, x):
        if self.training and self.sigma != 0:
            scale = self.sigma * x.std().detach() if self.is_relative_detach else self.sigma * x.std()
            sampled_noise = self.noise.repeat(*x.size()).float().normal_() * scale
            x = x + sampled_noise
        return x


class EncoderComponent(nn.Module):

    def __init__(self, cell_id_input_dim, num_gene=1):

        super(EncoderComponent, self).__init__()
        # self.cell_id_2 = nn.Linear(100, cell_id_emb_dim)
        self.cell_id_1 = nn.Linear(cell_id_input_dim, 200)
        self.cell_id_2 = nn.Linear(200, 100)
        self.cell_id_3 = nn.Linear(100, 50)
        self.cell_id_embed_linear_only = nn.Sequential(self.cell_id_1, nn.ReLU(),
                                                       self.cell_id_2, nn.ReLU(),
                                                       self.cell_id_3, nn.ReLU())

        self.cell_id_embed = nn.Sequential(nn.Linear(cell_id_input_dim, 200), nn.Linear(200, 50))
        self.trans_cell_embed_dim = 32
        # self.cell_id_embed_1 = nn.Linear(1, self.trans_cell_embed_dim)
        self.cell_id_transformer = nn.Transformer(d_model=self.trans_cell_embed_dim, nhead=4,
                                                  num_encoder_layers=1, num_decoder_layers=1,
                                                  dim_feedforward=self.trans_cell_embed_dim * 4)

        self.expand_to_num_gene = nn.Linear(50, 978)
        self.pos_encoder = PositionalEncoding(self.trans_cell_embed_dim)
        self.num_gene = num_gene

    def forward(self, cell_feature, linear_only=False):

        '''
        :cell_feature: tensor: batch * cell input dim (978)
        :epoch: int
        :linear_only: boolean: whether to include the transformer part
        :return: cell_id_embed: embeded cell hidden representation repeat self.num_gene times (batch * 978 * 50)
                : cell_hidden_: embeded cell hidden representation (batch * 50)
        '''
        if linear_only:
            cell_id_embed = self.cell_id_embed_linear_only(cell_feature)
            cell_id_embed = cell_id_embed.unsqueeze(1)
            cell_hidden_ = cell_id_embed.contiguous().view(cell_id_embed.size(0), -1)
            cell_id_embed = cell_id_embed.repeat(1, self.num_gene, 1)
        else:
            cell_id_embed = self.cell_id_embed(cell_feature)  # Transformer
            cell_id_embed = cell_id_embed.unsqueeze(-1)  # Transformer
            cell_id_embed = cell_id_embed.repeat(1, 1, self.trans_cell_embed_dim)
            cell_id_embed = self.pos_encoder(cell_id_embed)
            cell_id_embed = self.cell_id_transformer(cell_id_embed, cell_id_embed)  # Transformer
            cell_hidden_, _ = torch.max(cell_id_embed, -1)
            cell_id_embed = cell_hidden_.unsqueeze(1).repeat(1, self.num_gene, 1)
        return cell_id_embed, cell_hidden_


class CEAE(BaseAE):

    def __init__(self, input_dim: int, latent_dim: int = 50, **kwargs) -> None:
        super(CEAE, self).__init__()
        self.latent_dim = latent_dim

        self.encoder = EncoderComponent(cell_id_input_dim=input_dim)
        # build decoder
        hidden_dims = [100, 200]
        modules = []

        modules.append(
            nn.Sequential(
                nn.Linear(latent_dim, hidden_dims[0], bias=True),
                nn.SELU(),
            )
        )

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.Linear(hidden_dims[i], hidden_dims[i + 1], bias=True),
                    nn.SELU(),
                )
            )
        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
            nn.Linear(hidden_dims[-1], input_dim)
        )

    def encode(self, input: Tensor) -> Tensor:
        latent_code = self.encoder(input)[-1]
        return latent_code

    def decode(self, z: Tensor) -> Tensor:
        embed = self.decoder(z)
        outputs = self.final_layer(embed)

        return outputs

    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        z = self.encode(input)
        return [input, self.decode(z), z]

    def loss_function(self, *args, **kwargs) -> dict:
        input = args[0]
        recons = args[1]

        recons_loss = F.mse_loss(input, recons)
        loss = recons_loss

        return {'loss': loss, 'recons_loss': recons_loss}

    def sample(self, num_samples: int, current_device: int, **kwargs) -> Tensor:
        z = torch.randn(num_samples, self.latent_dim)
        z = z.to(current_device)
        samples = self.decode(z)

        return samples

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        return self.forward(x)[1]
