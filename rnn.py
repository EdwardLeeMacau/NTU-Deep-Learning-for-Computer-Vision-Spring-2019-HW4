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

from cnn import resnet50
import dataset
import utils
import predict

DEVICE = utils.selectDevice()

class LSTM_Net(nn.Module):
    """ The model to learn the video information by LSTM kernel """
    def __init__(self, feature_dim, hidden_dim, output_dim, num_layers=1, bias=True, batch_first=False, dropout=0, bidirectional=False, seq_predict=False):
        """
          Params:
          - feature_dim: dimension of the feature vector extracted from frames
          - hidden_dim: dimension of the hidden layer
        """
        super(LSTM_Net, self).__init__()

        self.feature_dim   = feature_dim
        self.hidden_dim    = hidden_dim
        self.output_dim    = output_dim
        self.num_layer     = num_layers
        self.bias          = bias
        self.batch_first   = batch_first
        self.dropout       = dropout
        self.bidirectional = bidirectional
        self.seq_predict   = seq_predict

        self.recurrent  = nn.LSTM(feature_dim, hidden_dim, num_layers=num_layers, bias=bias, batch_first=batch_first, dropout=dropout, bidirectional=bidirectional)
        self.fc_out     = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, output_dim),
        )
    def forward(self, seq_in):
        """
          Params:
          - seq_in: the tensor of frames
                    if batch_first: (batchsize, length, feature_dim)
                    if not:         (length, batchsize, feature_dim)
          
          Return:
          - out: the class prediction tensor: (batch, output_dim)
        """
        # -------------------------------------------------------------------
        # lstm_out, hidden_state, cell_state = LSTM(x, (hidden_state, cell_state))
        # -> lstm_out is the hidden_state tensor of the highest lstm cell.
        # -------------------------------------------------------------------
        lstm_out, (hidden_state, cell_state) = self.recurrent(seq_in)
        
        # get the last output of the model
        if self.seq_predict:
            raise NotImplementedError
        else:
            if self.batch_first:
                out = lstm_out[:,-1,:]
            else:
                out = lstm_out[-1]

        # through a linear layer to get the output
        out = self.fc_out(out)

        return out, (hidden_state, cell_state)

def main():
    torch.manual_seed(1)

    model = LSTM_Net(2048, 128, 11, num_layers=1, batch_first=True)

    # Way to initial the model parameters
    for param in model.recurrent.parameters():
        if len(param.shape) >= 2:
            nn.init.orthogonal_(param)
        

if __name__ == "__main__":
    main()
