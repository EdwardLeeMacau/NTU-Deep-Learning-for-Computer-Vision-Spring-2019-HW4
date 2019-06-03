"""
  FileName     [ tsne.py ]
  PackageName  [ HW4 ]
  Synopsis     [ HW4-1 t-SNE video feature visualiztion ]

  Library:
    scikit-video    1.1.11
    numpy           1.16.2
    ffmpeg
    ffprobe
"""

import argparse
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from sklearn.manifold import TSNE
from torch.autograd import Variable
from torch.utils.data import DataLoader

from dataset import TrimmedVideos
import utils
from rnn import LSTM_Net
from classifier import Classifier

parser = argparse.ArgumentParser()
parser.add_argument("--resume", default="./model/problem2.pth", type=str, help='The directory of model to load.')
parser.add_argument("--graph", default="./output", type=str, help='The folder to save the tsne figure')
parser.add_argument("--video", default="./hw4_data/TrimmedVideos/video/valid", type=str, help='The directory of the videos')
parser.add_argument("--label", default="./hw4_data/TrimmedVideos/label/valid", type=str, help='The directory of the ground truth label')
parser.add_argument("--feature", default="./hw4_data/TrimmedVideos/feature/valid" ,type=str, help='The path of the features')
parser.add_argument("--plot_num", default=1024, type=int, help='The number of points in the graphs')
parser.add_argument("--plot_size", default=20, type=int, help="The size of points in the graphs")
opt = parser.parse_args()

DEVICE = utils.selectDevice()

def dimension_reduction_cnn(fname, loader, model=None):
    if model:
        model.eval()

    with torch.no_grad():
        dataiter = iter(loader)

        features, labels = dataiter.next()
        print('labels.shape:   {}'.format(labels.shape))
        print('Features.shape: {}'.format(features.shape))

        features = features[:opt.plot_num]
        labels   = labels[:opt.plot_num]
        
        features_embedded = TSNE(n_components=2).fit_transform(features)

        print('features_embedded.shape: ', features_embedded.shape)
        print('plot_num: ', opt.plot_num )
        print('labels.shape :', labels.shape)

        plot_features(fname, features_embedded, labels, opt.plot_num)
    
    return

def dimension_reduction_rnn(fname, loader, model=None):
    if model:
        model.eval()

    with torch.no_grad():
        dataiter = iter(loader)

        features = np.zeros((opt.plot_num, 128), dtype=torch.float32)
        labels   = np.zeros(opt.plot_num, dtype=torch.float32)

        for index, (feature, label) in enumerate(loader, 0):
            if index == opt.plot_num: break

            feature = feature.to(DEVICE)
            feature = model(feature).cpu().detach().data.numpy()
            label   = label.cpu().detach().data.numpy()

            features[i] = feature
            labels[i]   = label
        
        embedded = TSNE(n_components=2).fit_transform(features)

        print('features_embedded.shape: ', features_embedded.shape)
        print('plot_num: ', opt.plot_num )
        print('labels.shape :', labels.shape)

        plot_features(fname, embedded, labels, opt.plot_num)
    
    return

def plot_features(fname, features, labels, plot_num, title=""):
    '''
      Params:
        features_embedded_mix   : tensor [n, 2],  target + source
        labels_mix              : tensor [n],  target + source
        plot_num_target         : index of the above 2 tensor, thres between target / source
        target_domain           : dataset name
        src_model_domain        : dataset name
    
      Return: None
    '''
    colors = plt.get_cmap('Set1')

    for num in range(11):
        mask_target = (labels == num)
        x = torch.masked_select( x, mask_target )
        y = torch.masked_select( y, mask_target )
        
        plt.scatter(x, y, s=opt.plot_size, c=colors(num), alpha=0.6, label=str(num))
        
    plt.title(title)
    plt.legend(loc=0)
    plt.imsave(fname)

    return

def main():
    valids_p1 = TrimmedVideos(None, opt.label, opt.feature_path, sample=4, transform=transforms.ToTensor())
    loader_p1 = DataLoader(valids_p1, batch_size=1024, shuffle=False)

    valids_p2 = TrimmedVideos(None, opt.label, opt.feature_path, downsample=12, transform=transforms.ToTensor())
    loader_p2 = DataLoader(valids_p2, batch_size=1, shuffle=False)

    recurrent = utils.loadModel(opt.resume, 
                    LSTM_Net(2048, 128, 11, num_layers=2, bias=True, 
                    dropout=0.2, bidirectional=False, seq_predict=False)
                ).to(DEVICE)

    graph_1 = os.path.join(opt.graph, 'p1_tsne.png')
    graph_2 = os.path.join(opt.graph, 'p2_tsne.png')

    dimension_reduction_cnn(graph_1, loader_p1)
    dimension_reduction_rnn(graph_2, loader_p2, recurrent)

if __name__ == '__main__':   
    main()
