"""
  FileName     [ classifier.py ]
  PackageName  [ HW4 ]
  Synopsis     [ Fully connected models for predict labels ]
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
        super(Classifier, self).__init__()
        
        self.fc = nn.Linear(feature_dim, num_class)
        self.bn = norm_layer(num_class)
        self.activation = activation

    def forward(self, x):
        x = self.fc(x)

        if self.bn:
            x = self.bn(x)

        if self.activation:
            x = self.activation(x)

        return x
