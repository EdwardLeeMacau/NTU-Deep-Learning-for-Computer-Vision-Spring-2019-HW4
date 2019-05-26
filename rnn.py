"""
  FileName     [ rnn.py ]
  PackageName  [ HW4 ]
  Synopsis     [ Test RNN, LSTM, bidirectional LSTM and GRU model ]
"""

import argparse
import datetime
import logging
import logging.config
import os
import random
from datetime import date

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from matplotlib import pyplot as plt
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from cnn import resnet50, Classifier
import dataset
import utils
import predict

DEVICE = utils.selectDevice()

def main():
    rnn  = nn.RNN()
    lstm = nn.LSTM()
    gru  = nn.GRU()

if __name__ == "__main__":
    main()