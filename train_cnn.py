"""
  FileName     [ train_cnn.py ]
  PackageName  [ HW4 ]
  Synopsis     [ CNN action recognition training methods ]
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
from classifier import Classifier
import dataset
import utils
import predict

# Set as true when the I/O shape of the model is fixed
cudnn.benchmark = True
DEVICE = utils.selectDevice()

parser = argparse.ArgumentParser()
# Basic Training setting
parser.add_argument("--epochs", type=int, default=50, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=1e-4, help="adam: learning rate")
parser.add_argument("--gamma", type=float, default=0.1, help="The ratio of decaying learning rate everytime")
parser.add_argument("--milestones", type=int, nargs='*', default=[10], help="Which epoch to decay the learning rate")
parser.add_argument("--weight_decay", type=float, default=1e-4, help="weight regularization")
parser.add_argument("--momentum", default=0.9, type=float, help="SGD Momentum, Default: 0.9")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--finetune", action="store_true", help="finetune the pretrained network")
parser.add_argument("--normalize", default=True, action="store_true", help="normalized the dataset images")
parser.add_argument("--activation", default="LeakyReLU", help="the activation function use at training")
# Message logging, model saving setting
parser.add_argument("--tag", type=str, help="tag for this training")
parser.add_argument("--checkpoints", type=str, help="path to save the checkpoints")
parser.add_argument("--step", type=int, default=1000, help="step to test the model performance")
parser.add_argument("--save_interval", type=int, default=10, help="interval epoch between everytime saving the model.")
parser.add_argument("--log_interval", type=int, default=10, help="interval between everytime logging the training status.")
parser.add_argument("--log", default="./log", help="the root directory to save the training details")
# Devices setting
parser.add_argument("--gpus", type=int, default=1, help="nums of gpu to use")
parser.add_argument("--cuda", default=True, help="Use cuda?")
parser.add_argument("--threads", type=int, default=8, help="number of cpu threads to use during batch generation")
# Load dataset, pretrain model setting
parser.add_argument("--resume", type=str, help="Path to checkpoint (default: none)")
parser.add_argument("--train", type=str, help="path to load train datasets")
parser.add_argument("--val", type=str, help="path to load val datasets")

opt = parser.parse_args()

def train(extractor, classifier, train_loader, val_loader, optim, epoch, criterion):
    """ Train each epoch with DANN framework. """
    trainloss = []
    trainaccs = []
    extractor.train()
    classifier.train()
    
    for index, (data, label) in enumerate(train_loader, 1):
        data, label = data.to(DEVICE), label.to(DEVICE)
        batchsize   = label.shape[0]
        
        #-----------------------------------------------------------------------------
        # Setup optimizer: clean the learning rate and set the learning rate (if need)
        #-----------------------------------------------------------------------------
        # optim = utils.set_optimizer_lr(optim, lr)
        optim.zero_grad()

        #---------------------------------------
        # Get features, class pred, domain pred:
        #---------------------------------------
        feature = extractor(data)
        predict = classifier(feature)

        #---------------------------
        # Compute the loss, accuracy
        #---------------------------
        loss = criterion(predict, label)
        loss.backward()
        optim.step()

        predict = predict.cpu().detach().numpy()
        label   = label.cpu().detach().numpy()
        acc     = np.mean(np.argmax(predict, axis=1) == label)

        trainloss.append(loss.item() * batchsize)
        trainaccs.append(acc * batchsize)

        if index % opt.log_interval == 0:
            print("[Epoch {}] [ {:4d}/{:4d} ] [acc: {:.2f}%] [loss: {:.4f}]".format(
                    epoch, index, len(train_loader), 100 * acc, loss.item()))

    return extractor, classifier, trainloss / len(train_loader.dataset), trainaccs / len(train_loader.dataset)

def single_frame_recognition():
    """ Using 2D CNN network to recognize the action. """
    #-----------------------------------------------------
    # Create Model, optimizer, scheduler, and loss function
    #------------------------------------------------------
    extractor  = resnet50(pretrained=True).to(DEVICE)
    classifier = Classifier(2048 * 14 * 14, 11).to(DEVICE)

    optimizer = optim.Adam(classifier.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2), weight_decay=opt.weight_decay)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=opt.milestones, gamma=opt.gamma)
    
    criterion = nn.CrossEntropyLoss().to(DEVICE)

    if opt.normalize:
        transform = transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.ToTensor(),
        ])
    else:
        transform = transforms.ToTensor()

    train_set  = dataset.TrimmedVideos(opt.train, train=True, transform=transform)
    val_set    = dataset.TrimmedVideos(opt.val, train=False, transform=transform)
    train_loader = DataLoader(train_set, batch_size=opt.batch_size, shuffle=True, num_workers=opt.threads)
    val_loader   = DataLoader(val_set, batch_size=opt.batch_size, shuffle=True, num_workers=opt.threads)

    print("Train: \t{}".format(len(train_set)))
    print("Val: \t{}".format(len(val_set)))
    
    #------------------
    # Train the models
    #------------------
    for epoch in range(1, opt.epochs + 1):
        scheduler.step()
        extractor, classifier = train(extractor, classifier, train_loader, val_loader, optimizer, epoch, criterion)
        accuracy, val_loss    = predict.val(extractor, classifier, val_loader, epoch, criterion)
        
        with open(os.path.join(opt.log, opt.tag, 'statistics.txt'), 'w') as textfile:
            pass
            
    return extractor, classifier

def draw_graphs(train_loss, val_loss, train_acc, val_acc, x, 
                loss_filename="loss.png", loss_log_filename="loss_log.png", acc_filename="acc.png", acc_log_filename="acc_log.png"):
    # Linear scale of loss curve
    plt.clf()
    plt.figure(figsize=(12.8, 7.2))
    plt.plot(x, train_loss, label="TrainLoss", color='b')
    plt.plot(x, val_loss, label="ValLoss", color='r')
    plt.plot(x, np.repeat(np.amin(val_loss), len(x)), ':')
    plt.legend(loc=0)
    plt.xlabel("Epoch(s)")
    plt.title("Loss vs Epochs")
    plt.savefig(os.path.join(opt.log, opt.tag, loss_filename))
    
    # Log scale of loss curve
    plt.yscale('log')
    plt.title("Loss vs Epochs")
    plt.savefig(os.path.join(opt.log, opt.tag, loss_log_filename))

    # Linear scale of accuracy curve
    plt.clf()
    plt.figure(figsize=(12.8, 7.2))
    plt.plot(x, train_acc, label="Train Acc", color='b')
    plt.plot(x, val_acc, label="Val Acc", color='r')
    plt.plot(x, np.repeat(np.amax(val_acc), len(x)), ':')
    plt.legend(loc=0)
    plt.xlabel("Epoch(s)")
    plt.title("Accuracy vs Epochs")
    plt.savefig(os.path.join(opt.log, opt.tag, acc_filename))
    
    # Log scale of accuracy curve
    plt.yscale('log')
    plt.title("Accuracy vs Epochs")
    plt.savefig(os.path.join(opt.log, opt.tag, acc_log_filename))
    return

def details(path):
    makedirs = []
    
    folder = os.path.dirname(path)
    while not os.path.exists(folder):
        makedirs.append(folder)
        folder = os.path.dirname(folder)

    while len(makedirs) > 0:
        makedirs, folder = makedirs[:-1], makedirs[-1]
        os.makedirs(folder)

    with open(path, "w") as textfile:
        for item, values in vars(opt).items():
            msg = "{:16} {}".format(item, values)
            print(msg)
            textfile.write(msg)

def main():
    """ Make the directory and check whether the dataset is exists """
    if not os.path.exists(opt.train):
        raise IOError("Path {} doesn't exist".format(opt.train))

    if not os.path.exists(opt.val):
        raise IOError("Path {} doesn't exist".format(opt.val))

    os.makedirs(opt.checkpoints, exist_ok=True)
    os.makedirs(os.path.join(opt.checkpoints, opt.tag), exist_ok=True)
    os.makedirs(opt.log, exist_ok=True)
    os.makedirs(os.path.join(opt.log, opt.tag), exist_ok=True)

    # Write down the training details (opt)
    details(os.path.join(opt.log, opt.tag, "train_setting.txt"))

    # Train the video recognition model with single frame (cnn) method
    single_frame_recognition()
    
    return

if __name__ == "__main__":
    os.system("clear")

    opt.tag = "{}_{}".format(opt.tag, date.today().strftime("%Y%m%d"))
    for key, value in vars(opt):
        print("{:15} {}".format(key, value))
    
    main()
