import torch
import torch.nn as nn
from types_ import *
from typing import List

class MultiEncoderDecoder(nn.Module):

    def __init__(self, decoder, encoders: List = None, normalize_flag=False):
        super(MultiEncoderDecoder, self).__init__()
        self.encoders = encoders
        self.decoder = decoder
        self.normalize_flag = normalize_flag

    def forward(self, input: Tensor) -> Tensor:
        encoded_input = self.encode(input)
        if self.normalize_flag:
            encoded_input = nn.functional.normalize(encoded_input, p=2, dim=1)
        output = self.decoder(encoded_input)
        return output

    def encode(self, input: Tensor) -> Tensor:
        latent_code = None
        inputs = torch.split(input, input.shape[-1]//len(self.encoders), dim=1)
        for i in range(len(self.encoders)):
            if i == 0:
                latent_code = self.encoders[i](inputs[i])
            else:
                latent_code = torch.cat((latent_code, self.encoders[i](inputs[i])), dim=1)
        return latent_code

    def decode(self, z: Tensor) -> Tensor:
        return self.decoder(z)
