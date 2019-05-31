"""
  FileName     [ predict.py ]
  PackageName  [ HW4 ]
  Synopsis     [ (...) ]
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

parser = argparse.ArgumentParser()

# Basic Training setting
parser.add_argument("--batch_size", type=int, default=8, help="size of the batches")
parser.add_argument("--downsample", default=4, type=int, help="the downsample ratio of the training data.")
# Model dimension setting
parser.add_argument("--layers", default=1, help="the number of the recurrent layers")
parser.add_argument("--bidirection", default=False, action="store_true", help="Use the bidirectional recurrent network")
parser.add_argument("--hidden_dim", default=128, help="the dimension of the RNN's hidden layer")
parser.add_argument("--output_dim", default=11, type=int, help="the number of the class to predict")
# Message logging, model saving setting
parser.add_argument("--checkpoints", default="/media/disk1/EdwardLee/video/checkpoint", type=str, help="path to save the checkpoints")
parser.add_argument("--log_interval", type=int, default=10, help="interval between everytime logging the training status.")
# Devices setting
parser.add_argument("--gpus", type=int, default=1, help="nums of gpu to use")
parser.add_argument("--cuda", default=True, help="Use cuda?")
parser.add_argument("--threads", type=int, default=8, help="number of cpu threads to use during batch generation")
# Load dataset, pretrain model setting
parser.add_argument("--train", default="./hw4_data/TrimmedVideos", type=str, help="path to load train datasets")
parser.add_argument("--val", default="./hw4_data/TrimmedVideos", type=str, help="path to load validation datasets")
parser.add_argument("--output", default="./output/problem_2/p2_pred.csv", help="The predict csvfile path.")

opt = parser.parse_args()

# Set as true when the I/O shape of the model is fixed
cudnn.benchmark = True
DEVICE = utils.selectDevice()

def predict(extractor, model, loader):
    """ Predict the model's performance. """
    extractor.eval()
    model.eval()

    pred_results = pd.DataFrame()

    #----------------------------
    # Calculate the accuracy, loss
    #----------------------------
    for index, (video, video_name) in enumerate(loader, 1):
        batchsize   = len(video_name)

        if index % opt.log_interval == 0:
            print("Predicting: {}".format(index * batchsize))

        video      = video.to(DEVICE)
        feature    = extractor(video).view(batchsize, -1)
        predict    = model(feature).argmax(dim=1).cpu().tolist()
        video_name = [name.split("/")[-1] for name in video_name]

        list_of_tuple = list(zip(video_name, predict))
        pred_result   = pd.DataFrame(list_of_tuple, columns=["Video_name", "Action_labels"])
        pred_results  = pd.concat((pred_results, pred_result), axis=0, ignore_index=True)

    return pred_results

def model_structure_unittest():
    extractor  = resnet50(pretrained=True).to(DEVICE)
    classifier = LSTM_Net(2048, 128, 11, num_layers=opt.layers, bidirectional=opt.bidirection).to(DEVICE)    
    criterion = torch.nn.CrossEntropyLoss().to(DEVICE)

    return

def main():
    #-------------------------
    # Construce the DANN model
    #-------------------------
    if opt.output is None:
        raise IOError("Please pass an outputpath argument with --output")

    extractor, classifier = utils.loadModel(opt.model, resnet50(pretrained=False), Classifier())
    extractor  = extractor.to(DEVICE)
    classifier = classifier.to(DEVICE)
    
    predict_set = dataset.TrimmedVideosPredict(opt.dataset, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]))
    print("Dataset: {}".format(len(predict_set)))
    predict_loader = DataLoader(predict_set, batch_size=opt.batch_size, shuffle=False, num_workers=opt.threads)
    
    # Predict
    pred_results = predict(extractor, classifier, predict_loader)
    
    pred_results.to_csv(opt.output, index=False)
    print("Output File have been written to {}".format(opt.output))

if __name__ == "__main__":
    os.system("clear")
    
    for key, value in vars(opt).items():
        print("{:15} {}".format(key, value))
    
    # check()
    main()
