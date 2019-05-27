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
parser.add_argument("--batch_size", type=int, default=16, help="size of the batches")
parser.add_argument("--lr", type=float, default=1e-4, help="adam: learning rate")
parser.add_argument("--gamma", type=float, default=0.1, help="The ratio of decaying learning rate")
parser.add_argument("--milestones", type=int, nargs='*', default=[10], help="The epoch to decay the learning rate")
parser.add_argument("--optimizer", type=str, default="Adam", help="The optimizer to use in this training")
parser.add_argument("--weight_decay", type=float, default=1e-4, help="weight regularization")
parser.add_argument("--momentum", default=0.9, type=float, help="SGD Momentum, Default: 0.9")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--finetune", action="store_true", help="finetune the pretrained network")
parser.add_argument("--normalize", default=True, action="store_true", help="normalized the dataset images")
# Model parameter setting
parser.add_argument("--activation", default="LeakyReLU", help="the activation function use at training")
parser.add_argument("--sample", default=4, type=int, help="the number of frames to catch")
parser.add_argument("--output_dim", default=11, type=int, help="the number of the class to predict")
# Message logging, model saving setting
parser.add_argument("--tag", type=str, help="tag for this training")
parser.add_argument("--checkpoints", default="/media/disk1/EdwardLee/video/checkpoint", type=str, help="path to save the checkpoints")
parser.add_argument("--step", type=int, default=1000, help="step to test the model performance")
parser.add_argument("--save_interval", type=int, default=1, help="interval epoch between everytime saving the model.")
parser.add_argument("--log_interval", type=int, default=10, help="interval between everytime logging the training status.")
parser.add_argument("--val_interval", type=int, default=100, help="interval between everytime validating the model performance.")
parser.add_argument("--log", default="./log", help="the root directory to save the training details")
# Devices setting
parser.add_argument("--gpus", type=int, default=1, help="nums of gpu to use")
parser.add_argument("--cuda", default=True, help="Use cuda?")
parser.add_argument("--threads", type=int, default=8, help="number of cpu threads to use during batch generation")
# Load dataset, pretrain model setting
parser.add_argument("--resume", type=str, help="Path to checkpoint (default: none)")
parser.add_argument("--train", default="./hw4_data/TrimmedVideos", type=str, help="path to load train datasets")
parser.add_argument("--val", default="./hw4_data/TrimmedVideos", type=str, help="path to load val datasets")

opt = parser.parse_args()

def train(extractor, classifier, train_loader, val_loader, optimizer, epoch, criterion):
    """ Train the classificaiton network. """
    trainloss = 0.0
    trainaccs = 0.0
    extractor.train()
    classifier.train()
    
    for index, (data, label) in enumerate(train_loader, 1):
        batchsize = label.shape[0]
        data, label = data.to(DEVICE).squeeze(0).view(-1, 3, 240, 320), label.to(DEVICE).view(-1)
        # print("Data.shape:  {}".format(data.shape))
        # print("Label.shape: {}".format(label.shape))
        # print(data)
        # print(data.dtype)
        
        #-----------------------------------------------------------------------------
        # Setup optimizer: clean the learning rate and set the learning rate (if need)
        #-----------------------------------------------------------------------------
        # optim = utils.set_optimizer_lr(optim, lr)
        optimizer.zero_grad()

        #---------------------------------------
        # Get features, class predict:
        #   data:          (batchsize * opt.sample, 3, 240, 320)
        #   feature:       (batchsize, 2048 * opt.sample)
        #   class predict: (batchsize, num_class)
        #---------------------------------------
        feature = extractor(data).view(batchsize, -1)
        # print("Feature.shape: {}".format(feature.shape))
        predict = classifier(feature)
        # print("Predict.shape: {}".format(predict.shape))

        #---------------------------
        # Compute the loss, accuracy
        #---------------------------
        loss = criterion(predict, label)
        loss.backward()
        optimizer.step()

        predict = predict.cpu().detach().numpy()
        label   = label.cpu().detach().numpy()
        acc     = np.mean(np.argmax(predict, axis=1) == label)

        trainloss += loss.item()
        trainaccs += acc

        if index % opt.log_interval == 0:
            print("[Epoch {}] [ {:4d}/{:4d} ] [acc: {:.2f}%] [loss: {:.4f}]".format(
                epoch, index, len(train_loader), 100 * trainaccs / index, trainloss / index
            ))

    # Print the average performance at the end time of each epoch
    trainloss = trainloss / len(train_loader)
    trainaccs = trainaccs / len(train_loader)
    print("[Epoch {}] [ {:4d}/{:4d} ] [acc: {:.2f}%] [loss: {:.4f}]".format(
        epoch, len(train_loader), len(train_loader), 100 * trainaccs, trainloss))

    return extractor, classifier, trainloss, trainaccs

def val(extractor, classifier, loader, epoch, criterion, log_interval=10):
    """ Validate the classificaiton network. """
    extractor.eval()
    classifier.eval()
    
    valaccs = 0.0
    valloss = 0.0
    
    #----------------------------
    # Calculate the accuracy, loss
    #----------------------------
    for index, (data, label) in enumerate(loader, 1):
        batchsize   = data.shape[0]
        
        data, label = data.view(-1, 3, 240, 320).to(DEVICE), label.type(torch.long).view(-1).to(DEVICE)
        
        feature = extractor(data).view(batchsize, -1)
        predict = classifier(feature)
        
        # loss
        loss = criterion(predict, label)
        valloss += loss.item()
        
        # Class Accuracy
        predict = predict.cpu().detach().numpy()
        label   = label.cpu().detach().numpy()
        acc     = np.mean(np.argmax(predict, axis=1) == label)
        valaccs += acc

        if index % log_interval == 0:
            print("[Epoch {}] [ {:4d}/{:4d} ]".format(epoch, index, len(loader)))

    valaccs = valaccs / len(loader)
    valloss = valloss / len(loader)
    print("[Epoch {}] [Validation] [ {:4d}/{:4d} ] [acc: {:.2f}%] [loss: {:.4f}]".format(epoch, len(loader), len(loader), 100 * valaccs, valloss))

    return valaccs, valloss

def single_frame_recognition():
    """ Using 2D CNN network to recognize the action. """
    #-----------------------------------------------------
    # Create Model, optimizer, scheduler, and loss function
    #------------------------------------------------------
    extractor  = resnet50(pretrained=True).to(DEVICE)
    classifier = Classifier(2048 * opt.sample, opt.output_dim).to(DEVICE)

    if opt.optimizer == "Adam":
        optimizer = optim.Adam(classifier.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2), weight_decay=opt.weight_decay)
    elif opt.optimizer == "SGD":
        optimizer = optim.SGD(classifier.parameters(), lr=opt.lr, momentum=opt.momentum, weight_decay=opt.weight_decay)
    else:
        raise NotImplementedError

    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=opt.milestones, gamma=opt.gamma)
    
    criterion = nn.CrossEntropyLoss().to(DEVICE)

    if opt.normalize:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    else:
        transform = transforms.ToTensor()

    train_set  = dataset.TrimmedVideos(opt.train, train=True, sample=4, transform=transform)
    val_set    = dataset.TrimmedVideos(opt.val, train=False, sample=4, transform=transform)
    train_loader = DataLoader(train_set, batch_size=opt.batch_size, shuffle=True, num_workers=opt.threads)
    val_loader   = DataLoader(val_set, batch_size=opt.batch_size, shuffle=True, num_workers=opt.threads)

    print("Train: \t{}".format(len(train_set)))
    print("Val: \t{}".format(len(val_set)))
    
    #------------------
    # Train the models
    #------------------
    trainloss = []
    trainaccs = []
    valloss   = []
    valaccs   = []
    epochs    = []
    for epoch in range(1, opt.epochs + 1):
        scheduler.step()
        
        # Save the train loss and train accuracy
        extractor, classifier, loss, acc = train(extractor, classifier, train_loader, val_loader, optimizer, epoch, criterion)
        trainloss.append(loss)
        trainaccs.append(acc)

        # Save the validation loss and validation accuracy
        acc, loss = val(extractor, classifier, val_loader, epoch, criterion)
        valloss.append(loss)
        valaccs.append(acc)

        # Save the epochs
        epochs.append(epoch)

        with open(os.path.join(opt.log, "problem_1", opt.tag, 'statistics.txt'), 'w') as textfile:
            textfile.write("\n".join(map(lambda x: str(x), (trainloss, trainaccs, valloss, valaccs, epochs))))

        if epoch % opt.save_interval == 0:
            savepath = os.path.join(opt.checkpoints, "problem_1", opt.tag, str(epoch) + '.pth')
            utils.saveCheckpoint(savepath, classifier, optimizer, scheduler, epoch)

            draw_graphs(trainloss, valloss, trainaccs, valaccs, epochs)
            
    return extractor, classifier

def draw_graphs(train_loss, val_loss, train_acc, val_acc, x, problem="problem_1",
                loss_filename="loss.png", loss_log_filename="loss_log.png", acc_filename="acc.png", acc_log_filename="acc_log.png"):
    # ----------------------------
    # Linear scale of loss curve
    # Log scale of loss curve
    # ----------------------------
    plt.clf()
    plt.figure(figsize=(12.8, 7.2))
    plt.plot(x, train_loss, label="TrainLoss", color='b')
    plt.plot(x, val_loss, label="ValLoss", color='r')
    plt.plot(x, np.repeat(np.amin(val_loss), len(x)), ':')
    plt.legend(loc=0)
    plt.xlabel("Epoch(s)")
    plt.title("Loss vs Epochs")
    plt.savefig(os.path.join(opt.log, problem, opt.tag, loss_filename))
    
    plt.yscale('log')
    plt.title("Loss vs Epochs")
    plt.savefig(os.path.join(opt.log, problem, opt.tag, loss_log_filename))

    # -------------------------------
    # Linear scale of accuracy curve
    # Log scale of accuracy curve
    # -------------------------------
    plt.clf()
    plt.figure(figsize=(12.8, 7.2))
    plt.plot(x, train_acc, label="Train Acc", color='b')
    plt.plot(x, val_acc, label="Val Acc", color='r')
    plt.plot(x, np.repeat(np.amax(val_acc), len(x)), ':')
    plt.legend(loc=0)
    plt.xlabel("Epoch(s)")
    plt.title("Accuracy vs Epochs")
    plt.savefig(os.path.join(opt.log, problem, opt.tag, acc_filename))
    
    plt.yscale('log')
    plt.title("Accuracy vs Epochs")
    plt.savefig(os.path.join(opt.log, problem, opt.tag, acc_log_filename))
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
    os.makedirs(os.path.join(opt.checkpoints, "problem_1"), exist_ok=True) 
    os.makedirs(os.path.join(opt.checkpoints, "problem_1", opt.tag), exist_ok=True)
    os.makedirs(opt.log, exist_ok=True)
    os.makedirs(os.path.join(opt.log, "problem_1"), exist_ok=True)
    os.makedirs(os.path.join(opt.log, "problem_1", opt.tag), exist_ok=True)

    # Write down the training details (opt)
    details(os.path.join(opt.log, opt.tag, "train_setting.txt"))

    # Train the video recognition model with single frame (cnn) method
    single_frame_recognition()
    
    return

if __name__ == "__main__":
    os.system("clear")    
    main()
