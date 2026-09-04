"""
The Afterimage autoencoder: a convolutional encoder/decoder pair trained on
general-purpose photos. Only the decoder is ever shipped to the browser --
see export_decoder.py and ../CLAUDE.md for why.

Encoder: 96x96x3 -> four stride-2 convs down to 6x6x256 -> dense to a
256-number latent vector.
Decoder: mirrors that back out, latent -> dense -> 6x6x256 -> four stride-2
transposed convs up to 96x96x3.
"""
import torch.nn as nn
import torch

RES = 96
CH = 3
LATENT = 256


class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(CH, 32, 4, stride=2, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 4, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 4, stride=2, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 4, stride=2, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
        )
        self.enc_fc = nn.Linear(256 * 6 * 6, LATENT)

        self.dec_fc = nn.Linear(LATENT, 256 * 6 * 6)
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, CH, 4, stride=2, padding=1),
        )

    def encode(self, x):
        h = self.enc(x)
        h = h.view(h.size(0), -1)
        return self.enc_fc(h)

    def decode(self, z):
        h = self.dec_fc(z)
        h = h.view(h.size(0), 256, 6, 6)
        return torch.sigmoid(self.dec(h))

    def forward(self, x):
        return self.decode(self.encode(x))
