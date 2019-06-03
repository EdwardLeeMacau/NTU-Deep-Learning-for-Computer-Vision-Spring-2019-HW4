"""
  FileName     [ predict_cnn.py ]
  PackageName  [ HW4 ]
  Synopsis     [  ]
"""

import argparse
import logging
import logging.config
import os
import random
import time

import numpy as np
import pandas as pd
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from matplotlib import pyplot as plt
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import dataset
import utils
from cnn import resnet50
from rnn import LSTM_Net
from classifier import Classifier

parser = argparse.ArgumentParser()

# Basic Training setting
parser.add_argument("--batch_size", type=int, default=4, help="size of the batches")
parser.add_argument("--downsample", default=12, type=int, help="the downsample ratio of the training data.")
# Model dimension setting
parser.add_argument("--output_dim", default=11, type=int, help="the number of the class to predict")
# Message logging, model saving setting
parser.add_argument("--log_interval", type=int, default=1, help="interval between everytime logging the training status.")
# Devices setting
parser.add_argument("--threads", type=int, default=8, help="number of cpu threads to use during batch generation")
# Load dataset, pretrain model setting
parser.add_argument("--video", default="./hw4_data/TrimmedVideos/video/train", type=str, help="path to the videos directory")
parser.add_argument("--label", default="./hw4_data/TrimmedVideos/label/gt_train.csv", type=str, help="path of the label csv file")
parser.add_argument("--output", required=True, default="./output", help="The predict csvfile path.")

opt = parser.parse_args()

opt.output = os.path.join(opt.output, 'p1_valid.txt')

# Set as true when the I/O shape of the model is fixed
cudnn.benchmark = True
DEVICE = utils.selectDevice()

def predict(extractor: nn.Module, model: nn.Module, loader: DataLoader) -> np.array:
    """ Predict the model's performance. """
    extractor.eval()
    model.eval()

    result = []
    
    for index, (video, _, video_name) in enumerate(loader, 1):
        batchsize = len(video_name)

        video      = video.to(DEVICE)
        feature    = extractor(video).view(batchsize, -1)
        predict    = model(feature).argmax(dim=1).cpu().tolist()
        video_name = [name.split("/")[-1] for name in video_name]

        result.append(*predict)

        if index % opt.log_interval == 0:
            print("[ {:4d}/{:4d} ]".format(len(index), len(loader)))

    return np.array(result).astype(int).transpose()

def main():
    extractor, classifier = utils.loadModel(opt.model, resnet50(pretrained=False), Classifier(8192, 11))
    extractor, classifier = extractor.to(DEVICE), classifier.to(DEVICE)
    
    predict_set = dataset.TrimmedVideos(opt.video_path, opt.label_path, None, sample=4, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]))
    print("Dataset: {}".format(len(predict_set)))
    predict_loader = DataLoader(predict_set, batch_size=opt.batch_size, shuffle=False, num_workers=opt.threads)
    
    # Predict
    results = predict(extractor, extractor, predict_loader)
    np.savetxt(opt.output, results, fmt='%d')
    print("Output File have been written to {}".format(opt.output))

if __name__ == "__main__":
    os.system("clear")
    
    for key, value in vars(opt).items():
        print("{:15} {}".format(key, value))
    
    main()
