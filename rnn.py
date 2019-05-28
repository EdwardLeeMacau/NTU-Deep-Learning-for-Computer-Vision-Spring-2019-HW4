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
from torch.nn.utils.rnn import pad_packed_sequence
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

    def forward(self, x):
        """
          Params:
          - x: the tensor of frames
               if batch_first: (batchsize, length, feature_dim)
               if not:         (length, batchsize, feature_dim)
          
          Return:
          - x: the class prediction tensor: 
               if seq_predict: (length, batch, output_dim)
               if not:         (batch, output_dim)
        """
        # -------------------------------------------------------------------
        # lstm_out, hidden_state, cell_state = LSTM(x, (hidden_state, cell_state))
        # -> lstm_out is the hidden_state tensor of the highest lstm cell.
        # -------------------------------------------------------------------
        x, _       = self.recurrent(x)
        x, seq_len = pad_packed_sequence(x, batch_first=self.batch_first)
        
        # get the output per frame of the model
        if self.seq_predict:
            batchsize = x.shape[0]
            x = x.view(-1, self.hidden_dim)
            x = self.fc_out(x)
            x = x.view(-1, batchsize, self.output_dim)
            
            return x, seq_len
        
        # get the last 1 output if only it is needed.
        if self.batch_first:
            x = x[:,-1]
        else:
            x = x[-1]

        x = self.fc_out(x)

        return x, seq_len

def main():
    torch.manual_seed(1)

    model = LSTM_Net(2048, 128, 11, num_layers=1, batch_first=False)

    # Way to initial the model parameters
    for param in model.recurrent.parameters():
        if len(param.shape) >= 2:
            nn.init.orthogonal_(param)
        

if __name__ == "__main__":
    main()
