from torch import nn
import numpy as np
import torch


class GedCompLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, v, D, c, normalize_factor=1.):
        return (.5 * v.T @ D @ v + c.T @ v)/normalize_factor
