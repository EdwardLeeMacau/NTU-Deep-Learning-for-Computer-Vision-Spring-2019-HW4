"""
  FileName     [ cnn.py ]
  PackageName  [ HW4 ]
  Synopsis     [ ResNet models for feature extracting ]

  * Feature extracting models:
    Resnet (Resnet18, Resnet34, Resnet50, Resnet101, Resnet152)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Function
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils import model_zoo

import utils

class Classifier(nn.Module):
    def __init__(self, feature_dim, num_class=11, norm_layer=nn.BatchNorm1d, activation=nn.ReLU(inplace=True)):
        self.fc = nn.Linear(feature_dim, num_class)
        self.bn = norm_layer(num_class)
        self.activation = activation

    def forward(self, x):
        x = self.fc(x)
        x = self.bn(x)

        if self.activation:
            x = self.activation(x)
        
        return x