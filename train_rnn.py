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
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence, pad_sequence

import dataset
import predict
import utils
from classifier import Classifier
from cnn import resnet50
from rnn import LSTM_Net

# Set as true when the I/O shape of the model is fixed
# cudnn.benchmark = True
DEVICE = utils.selectDevice()

parser = argparse.ArgumentParser()
# Basic Training setting
parser.add_argument("--epochs", type=int, default=50, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=8, help="size of the batches")
parser.add_argument("--lr", type=float, default=1e-4, help="adam: learning rate")
parser.add_argument("--gamma", type=float, default=0.1, help="The ratio of decaying learning rate")
parser.add_argument("--milestones", type=int, nargs='*', default=[10], help="The epoch to decay the learning rate")
parser.add_argument("--optimizer", type=str, default="Adam", help="The optimizer to use in this training")
parser.add_argument("--weight_decay", type=float, default=1e-4, help="weight regularization")
parser.add_argument("--momentum", default=0.9, type=float, help="SGD Momentum, Default: 0.9")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--dropout", default=0, help="the dropout probability of the recurrent network")
parser.add_argument("--finetune", action="store_true", help="finetune the pretrained network")
parser.add_argument("--normalize", default=True, action="store_true", help="normalized the dataset images")
parser.add_argument("--downsample", default=4, type=int, help="the downsample ratio of the training data.")
# Model dimension setting
parser.add_argument("--activation", default="LeakyReLU", help="the activation function use at training")
parser.add_argument("--layers", default=1, help="the number of the recurrent layers")
parser.add_argument("--bidirection", default=False, action="store_true", help="Use the bidirectional recurrent network")
parser.add_argument("--hidden_dim", default=128, help="the dimension of the RNN's hidden layer")
parser.add_argument("--output_dim", default=11, type=int, help="the number of the class to predict")
# Model parameter initialization setting
parser.add_argument("--init", type=str, help="define the network parameter initialization methods")
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
parser.add_argument("--feature", action='store_true', help='If true, use the preprocressed feature files instead of the images')
parser.add_argument("--train", default="./hw4_data/TrimmedVideos", type=str, help="path to load train datasets")
parser.add_argument("--val", default="./hw4_data/TrimmedVideos", type=str, help="path to load val datasets")

opt = parser.parse_args()

def train(extractor, recurrent, train_loader, val_loader, optimizer, epoch, criterion):
    """ Train the classificaiton network. """
    trainloss = 0.0
    trainaccs = 0.0
    extractor.train()
    recurrent.train()
    
    for index, (data, label, seq_len) in enumerate(train_loader, 1):
        batchsize = label.shape[0]
        
        data, label = data.to(DEVICE), label.to(DEVICE)
        print("Data.shape: {}".format(data.shape))
        print("Label.shape: {}".format(label.shape))

        #-----------------------------------------------------------------------------
        # Setup optimizer: clean the learning rate and set the learning rate (if need)
        #-----------------------------------------------------------------------------
        # optim = utils.set_optimizer_lr(optim, lr)
        optimizer.zero_grad()

        #---------------------------------------
        # Get features, class predict:
        #   data:          (batchsize, frames, 3, 240, 320)
        #   feature:       (batchsize, frames, 2048)
        #   class predict: (batchsize, num_class)
        #---------------------------------------
        feature = torch.Tensor([extractor(data[i]).view(batchsize, -1) for i in range(batchsize)])
        feature = pad_sequence(feature, batch_first=False)
        print("Feature.shape: {}".format(feature.shape))

        feature = pack_padded_sequence(feature, seq_len, batch_first=False)
        predict, _ = pad_packed_sequence(recurrent(feature), batch_first=False)
        print("Predict.shape: {}".format(predict.shape))
        print(predict)

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

    return extractor, recurrent, trainloss, trainaccs

def val(extractor, recurrent, loader, epoch, criterion, log_interval=10):
    """ Validate the recurrent network. """
    extractor.eval()
    recurrent.eval()

    valaccs = 0.0
    valloss = 0.0
    
    #----------------------------
    # Calculate the accuracy, loss
    #----------------------------
    for index, (data, label) in enumerate(loader, 1):
        batchsize   = label.shape[0]
        
        data, label = data.to(DEVICE), label.type(torch.long).view(-1).to(DEVICE)
        
        feature = extractor(data).view(batchsize, -1)
        predict, _ = pad_packed_sequence(recurrent(feature), batch_first=False)

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

def continuous_frame_recognition():
    """ Using RNN network to recognize the action. """
    start_epoch = 1

    #-----------------------------------------------------
    # Create Model, optimizer, scheduler, and loss function
    #------------------------------------------------------
    extractor  = resnet50(pretrained=True).to(DEVICE)
    recurrent  = LSTM_Net(2048, opt.hidden_dim, opt.output_dim, 
                        num_layers=opt.layers, bias=True, batch_first=False, dropout=opt.dropout, 
                        bidirectional=opt.bidirection, seq_predict=False).to(DEVICE)

    # Set optimizer
    if opt.optimizer == "Adam":
        optimizer = optim.Adam(recurrent.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2), weight_decay=opt.weight_decay)
    elif opt.optimizer == "SGD":
        optimizer = optim.SGD(recurrent.parameters(), lr=opt.lr, momentum=opt.momentum, weight_decay=opt.weight_decay)
    else:
        raise argparse.ArgumentError
        
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=opt.milestones, gamma=opt.gamma)
    
    # Load parameter
    if opt.resume:
        recurrent, optimizer, start_epoch, scheduler = utils.loadCheckpoint(opt.resume, extractor, recurrent, optimizer, scheduler, pretrained=(not opt.finetune))

    # Set criterion
    criterion = nn.CrossEntropyLoss().to(DEVICE)

    # Set dataloader
    if opt.normalize:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    else:
        transform = transforms.ToTensor()

    train_set  = dataset.TrimmedVideos(opt.train, train=True, downsample=opt.downsample, transform=transform)
    val_set    = dataset.TrimmedVideos(opt.val, train=False, downsample=opt.downsample, transform=transform)
    train_loader = DataLoader(train_set, batch_size=opt.batch_size, shuffle=True, collate_fn=utils.collate_fn, num_workers=opt.threads)
    val_loader   = DataLoader(val_set, batch_size=opt.batch_size, shuffle=True, collate_fn=utils.collate_fn, num_workers=opt.threads)

    # Show the memory used by neural network
    print("The neural network allocated GPU with {:.1f} MB".format(torch.cuda.memory_allocated() / 1024 / 1024))

    #------------------
    # Train the models
    #------------------
    trainloss = []
    trainaccs = []
    valloss   = []
    valaccs   = []
    epochs    = []

    for epoch in range(start_epoch, opt.epochs + 1):
        scheduler.step()
        
        # Save the train loss and train accuracy
        extractor, recurrent, loss, acc = train(extractor, recurrent, train_loader, val_loader, optimizer, epoch, criterion)
        trainloss.append(loss)
        trainaccs.append(acc)

        # Save the validation loss and validation accuracy
        acc, loss = val(extractor, recurrent, val_loader, epoch, criterion)
        valloss.append(loss)
        valaccs.append(acc)

        # Save the epochs
        epochs.append(epoch)

        with open(os.path.join(opt.log, opt.tag, 'statistics.txt'), 'w') as textfile:
            textfile.write("\n".join(map(lambda x: str(x), (trainloss, trainaccs, valloss, valaccs, epochs))))

        if epoch % opt.save_interval == 0:
            savepath = os.path.join(opt.checkpoints, "problem_2", opt.tag, str(epoch) + '.pth')
            utils.saveCheckpoint(savepath, extractor, recurrent, optimizer, scheduler, epoch)

            draw_graphs(trainloss, valloss, trainaccs, valaccs, epochs)
            
    return extractor, recurrent

def draw_graphs(train_loss, val_loss, train_acc, val_acc, x, problem="problem_2",
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

    if opt.feature and opt.finetune:
        raise argparse.ArgumentError

    os.makedirs(opt.checkpoints, exist_ok=True)
    os.makedirs(os.path.join(opt.checkpoints, "problem_2"), exist_ok=True) 
    os.makedirs(os.path.join(opt.checkpoints, "problem_2", opt.tag), exist_ok=True)
    os.makedirs(opt.log, exist_ok=True)
    os.makedirs(os.path.join(opt.log, "problem_2"), exist_ok=True)
    os.makedirs(os.path.join(opt.log, "problem_2", opt.tag), exist_ok=True)

    # Write down the training details (opt)
    details(os.path.join(opt.log, "problem_2", opt.tag, "train_setting.txt"))

    # Train the video recognition model with single frame (cnn) method
    continuous_frame_recognition()
    
    return

if __name__ == "__main__":
    os.system("clear")    
    main()
