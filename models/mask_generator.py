import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np
import pytorch_lightning as pl
import lightly
from loss import BarlowTwinsLoss
import sys
import copy
import os

from utils import knn_predict, BenchmarkModule
from lightly.models.simsiam import _prediction_mlp, _projection_mlp, SimSiam


class FeatureMask(pl.LightningModule):
    def __init__(self, projection_out_dim, enable_sigmoid):
        super().__init__()
        init_value = 1.0
        self.enable_sigmoid =enable_sigmoid
        if self.enable_sigmoid:
            init_value = 0.0
        self.mask = nn.Parameter(torch.ones(int(projection_out_dim)) * init_value)

    def forward(self):
        if self.enable_sigmoid:
            return torch.sigmoid(torch.ones_like(self.mask) * self.mask)
        return torch.ones_like(self.mask) * self.mask

