"""
  FileName     [ utils.py ]
  PackageName  [ HW4 ]
  Synopsis     [ Utility functions in package HW4 ]
"""

import sys

import numpy as np
import torch
from torch import nn, optim
# from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence

trimmedVideos_feature    = []
fullLengthVideos_feature = []

def set_optimizer_lr(optimizer, lr):
    """ set the learning rate in an optimizer, without rebuilding the whole optimizer """
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    
    return optimizer

def collate_fn(batch):
    """
      To define a function that reads the video by batch.
 
      Params:
      - batch: 
          In pytorch, dataloader generate batch of traindata by this way:
            `self.collate_fn([self.dataset[i] for i in indices])`
          
          In here, batch means `[self.dataset[i] for i in indices]`
          It's a list contains (datas, labels)

      Return:
      - batch
    """
    # ---------------------------------
    # batch[i][j]
    #   the type of batch[i] is tuple
    # 
    #   i=(0, size) means the batchsize
    #   j=(0, 1) means the data / label
    # ---------------------------------
    
    # Sorted the batch with the video length with the descending order
    batch   = sorted(batch, key=lambda x: x[0].shape[0], reverse=True)
    seq_len = [x[0].shape[0] for x in batch]
    label   = torch.cat([x[1].unsqueeze(0) for x in batch], dim=0)
    batch   = pad_sequence([x[0] for x in batch], batch_first=False)
    
    return (batch, label, seq_len)

def selectDevice():
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    return device

def saveCheckpoint(checkpoint_path, feature: nn.Module, model: nn.Module, optimizer: optim, scheduler: optim.lr_scheduler.MultiStepLR, epoch, pretrained=True):
    state = {
        'state_dict': model.state_dict(),
        'optimizer' : optimizer.state_dict(),
        'epoch': epoch,
        'scheduler': scheduler.state_dict()
    }

    if not pretrained:
        state['feature'] = feature.state_dict()

    torch.save(state, checkpoint_path)

    return

def loadCheckpoint(checkpoint_path: str, feature: nn.Module, model: nn.Module, optimizer: optim, scheduler: optim.lr_scheduler.MultiStepLR, pretrained=True):
    state = torch.load(checkpoint_path)

    resume_epoch = state['epoch']
    model.load_state_dict(state['state_dict'])
    optimizer.load_state_dict(state['optimizer'])
    scheduler.load_state_dict(state['scheduler'])

    if not pretrained:
        feature.load_state_dict(state['feature'])

    print('model loaded from %s' % checkpoint_path)

    return model, optimizer, resume_epoch, scheduler

def saveModel(checkpoint_path: str, feature:nn.Module, model: nn.Module, pretrained=True):
    """
      Params:
      - checkpoint_path: the directory of the model parameter
      - feature: the structure of the feature extractor
      - model: If cnn -> classifier
               If rnn -> recurrent
      - pretrained: If True, ignore the feature extractor pretrained model
                    If False, save the model parameter to the file
    """
    state = {'state_dict': model.state_dict()}

    if not pretrained:
        state['feature'] = feature.state_dict()

    torch.save(state, checkpoint_path)
    print('model saved to %s' % checkpoint_path)

def loadModel(checkpoint_path: str, feature: nn.Module, model: nn.Module, pretrained=True):
    """
      Params:
      - checkpoint_path: the directory of the model parameter
      - feature: the structure of the feature extractor
      - model: If cnn -> classifier
               If rnn -> recurrent
      - pretrained: If True, load the feature extractor pretrained model
                    If False, load the model parameter from the saved file
    """
    state = torch.load(checkpoint_path)
    model.load_state_dict(state['state_dict'])

    if not pretrained:
        feature.load_state_dict(state['feature'])

    print('Model loaded from %s' % checkpoint_path)

    return feature, model

def checkpointToModel(checkpoint_path: str, model_path: str):
    state = torch.load(checkpoint_path)

    newState = {
        'state_dict': state['state_dict']
    }

    torch.save(newState, model_path)
