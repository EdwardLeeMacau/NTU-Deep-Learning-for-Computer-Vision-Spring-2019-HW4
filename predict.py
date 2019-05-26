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
from cnn import resnet50, Classifier

parser = argparse.ArgumentParser()
# Basic Training setting
parser.add_argument("--batch_size", type=int, default=64, help="Images to read for every iteration")
parser.add_argument("--normalize", default=False, action="store_true", help="normalized the dataset images")
parser.add_argument("--activation", default="LeakyReLU", help="the activation function use at training")
# Message logging, model saving setting
parser.add_argument("--tag", type=str, help="tag for this training")
parser.add_argument("--checkpoints", type=str, help="path to load the checkpoints")
# Devices setting
parser.add_argument("--gpus", type=int, default=1, help="nums of gpu to use")
parser.add_argument("--threads", type=int, default=8, help="Number of cpu threads to use during batch generation")
parser.add_argument("--cuda", default=True, help="Use cuda?")
# Load dataset, pretrain model setting
parser.add_argument("--dataset", type=str, help="The root of input dataset")
parser.add_argument("--val", type=str, help="path to load val datasets")
parser.add_argument("--detail", default="./train_details", help="the root directory to save the training details")
# Saving prediction setting
parser.add_argument("--output", default="./output/dann/svhn_pred.csv", help="The predict csvfile path.")

opt = parser.parse_args()

# Set as true when the I/O shape of the model is fixed
cudnn.benchmark = True
DEVICE = utils.selectDevice()

def val(extractor, classifier, loader, epoch, criterion):
    extractor.eval()
    classifier.eval()
    
    total_accs = 0
    total_loss = 0
    
    #----------------------------
    # Calculate the accuracy, loss
    #----------------------------
    for _, (data, label) in enumerate(loader, 1):
        batchsize   = data.shape[0]
        
        data, label = data.to(DEVICE), label.type(torch.long).view(-1).to(DEVICE)
        
        feature = extractor(data).view(batchsize, -1)
        predict = classifier(feature)
        
        # loss
        loss = criterion(predict, label)
        total_loss += (loss.item() * batchsize)
        
        # Class Accuracy
        predict = predict.cpu().detach().numpy()
        label   = label.cpu().detach().numpy()
        acc     = np.mean(np.argmax(predict, axis=1) == label)
        total_accs += (acc * batchsize)

    acc  = total_accs / len(loader.dataset)
    loss = total_loss / len(loader.dataset)

    return acc, loss

def predict(extractor, classifier, loader):
    extractor.eval()
    classifier.eval()

    pred_results = pd.DataFrame()

    #----------------------------
    # Calculate the accuracy, loss
    #----------------------------
    for index, (video, video_name) in enumerate(loader, 1):
        batchsize   = len(img)

        if index % opt.log_interval == 0:
            print("Predicting: {}".format(index * batchsize))

        video      = video.to(DEVICE)
        feature    = extractor(video).view(batchsize, -1)
        predict    = classifier(feature).argmax(dim=1).cpu().tolist()
        video_name = [name.split("/")[-1] for name in video_name]

        list_of_tuple = list(zip(img_name, class_pred))
        pred_result   = pd.DataFrame(list_of_tuple, columns=["image_name", "label"])
        pred_results  = pd.concat((pred_results, pred_result), axis=0, ignore_index=True)

    return pred_results

def check():
    #-------------------------
    # Construce the DANN model
    #-------------------------
    extractor  = resnet50(pretrained=True).to(DEVICE)
    classifier = Classifier(2048 * 14 * 14, 11).to(DEVICE)    
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
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.ToTensor(),
    ]))
    print("Dataset: {}".format(len(predict_set)))
    predict_loader = DataLoader(predict_set, batch_size=opt.batch_size, shuffle=False, num_workers=opt.threads)
    
    # Predict
    pred_results = predict(extractor, classifier, predict_loader)
    
    pred_results.to_csv(opt.output, index=False)
    print("Output File have been written to {}".format(opt.output))

if __name__ == "__main__":
    os.system("clear")
    
    for key, value in vars(opt):
        print("{:15} {}".format(key, value))
    
    # check()
    main()
