import os
import sys
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50
import torch.nn.init as init


class HNet(nn.Module):
    # xlin: H matrix
    def __init__(self, ndata, dim=128):
        super().__init__()
        self.features = nn.Parameter(torch.randn(ndata, dim))

    def forward(self, idxs):
        return self.features[idxs, :]


class dgdtNet(nn.Module):
    # xlin: degradation module
    def __init__(self, hidden=216, hdim=128, outdim=256):
        super().__init__()
        self.fc1 = nn.Linear(hdim, hidden)
        self.bn = nn.BatchNorm1d(hidden)
        # self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, outdim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn(x)
        x = self.relu(x)
        # x = self.fc2(x)
        # x = self.relu(x)
        x = self.fc3(x)
        return x


class IntactNet(nn.Module):
    # xlin: may not use
    def __init__(self, ndata, hidden=216, hdim=128):
        super().__init__()
        self.H = HNet(ndata, hdim)
        self.dgdt1 = dgdtNet(hidden, hdim, hdim)
        self.dgdt2 = dgdtNet(hidden, hdim, hdim)

    def forward(self, idxs):
        x = self.H(idxs)
        img_feat = self.dgdt1(x)
        pnt_feat = self.dgdt2(x)
        return img_feat, pnt_feat
