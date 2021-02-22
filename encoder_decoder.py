import torch.nn as nn
from types_ import *
from gradient_reversal import GradientReversal

class EncoderDecoder(nn.Module):

    def __init__(self, encoder, decoder, normalize_flag=False,gr_flag=False):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.normalize_flag = normalize_flag
        self.gr_flag = gr_flag

    def forward(self, input: Tensor) -> Tensor:
        encoded_input = self.encoder(input)
        if self.normalize_flag:
            encoded_input = nn.functional.normalize(encoded_input, p=2, dim=1)
        if self.gr_flag:
            self.decoder = nn.Sequential(
                GradientReversal(),
                self.decoder
            )
        output = self.decoder(encoded_input)

        return output

    def encode(self, input: Tensor) -> Tensor:
        return self.encoder(input)

    def decode(self, z: Tensor) -> Tensor:
        return self.decoder(z)
