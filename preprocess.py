"""
  FileName     [ preprocess.py ]
  PackageName  [ HW4 ]
  Synopsis     [ Dataset of the HW4 ]
"""

import csv
import os
import random
import time

import numpy as np
import torch
import torchvision.transforms as transforms
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader, Dataset

import dataset
import utils
from cnn import resnet50

device = utils.selectDevice()

def video_to_features(data_path):
    """ Transfer the training set and validation set videos into features """

    # -----------------------------------------------------------------
    # To save the numpy array into the file, there are several options
    #   Machine readable:
    #   - ndarray.dump(), ndarray.dumps(), pickle.dump(), pickle.dumps():
    #       Generate .pkl file.
    #   - np.save(), np.savez(), np.savez_compressed()
    #       Generate .npy file
    #   - np.savetxt()
    #       Generate .txt file.
    # -----------------------------------------------------------------
    
    for train in (True, False):
        dataset = dataset.TrimmedVideos(data_path, train=train, downsample=1, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]))

        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    
        extractor = resnet50(pretrained=True).to(device).eval()

        if train:   train_val = "train"
        else:       train_val = "valid"
    
        for index, (data, _, category, name) in enumerate(dataloader, 1):
            data   = data.squeeze(0)
            datas  = np.zeros((data.shape[0], 2048), dtype=np.float)
            remain = data.shape[0]
            finish = 0

            while remain > 0:
                step = min(remain, 50)
                todo = data[finish : finish + step].to(device)
                datas[finish : finish + step] = extractor(todo).cpu().data.numpy()
                
                remain -= step
                finish += step

            print("{:4d} {:16d} {}".format(
                index, datas.shape, os.path.join(data_path, "feature", train_val, category[0], name[0] + ".npy")))

            # ------------------------------------
            # Save the feature tensor in .npy file
            # ------------------------------------
            if not os.path.exists(os.path.join(data_path, "feature", train_val, category[0])):
                os.makedirs(os.path.join(data_path, "feature", train_val, category[0]))

            np.savetxt(os.path.join(data_path, "feature", train_val, category[0], name[0] + ".npy"), datas, delimiter=',')

    return

if __name__ == "__main__":
    video_to_features("./hw4_data/TrimmedVideos")