"""U-Net with the same layout as version 1: four encoder blocks, a bottleneck and
four decoder blocks with transposed convolutions and skip connections.

Version 1 used 64 base channels (31.0 million parameters) at 256 by 256 on a
Colab GPU. Version 2 trains on CPU runners, so `base` is a parameter; the
experiment uses the same layout with fewer channels (see train.py).
"""
from __future__ import annotations

import torch
from torch import nn


class ConvBlock(nn.Sequential):
    def __init__(self, cin: int, cout: int):
        super().__init__(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        )


class UNet(nn.Module):
    def __init__(self, base: int = 64):
        super().__init__()
        ch = [base, base * 2, base * 4, base * 8]
        self.down = nn.ModuleList()
        cin = 3
        for c in ch:
            self.down.append(ConvBlock(cin, c))
            cin = c
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = ConvBlock(ch[-1], ch[-1] * 2)
        self.up = nn.ModuleList()
        self.dec = nn.ModuleList()
        cin = ch[-1] * 2
        for c in reversed(ch):
            self.up.append(nn.ConvTranspose2d(cin, c, 2, stride=2))
            self.dec.append(ConvBlock(c * 2, c))
            cin = c
        self.head = nn.Conv2d(ch[0], 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        for block in self.down:
            x = block(x)
            skips.append(x)
            x = self.pool(x)
        x = self.bottleneck(x)
        for up, dec, skip in zip(self.up, self.dec, reversed(skips), strict=True):
            x = dec(torch.cat([up(x), skip], dim=1))
        return self.head(x)


def n_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
