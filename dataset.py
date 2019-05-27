"""
  FileName     [ dataset.py ]
  PackageName  [ HW4 ]
  Synopsis     [ Dataset of the HW4 ]

  - Dataset:
    TrimmedVideos:              Prob 1, 2
    TrimmedVideosPredict:       Prob 1, 2
    FullLengthVideos:           Prob 3
    FullLengthVideosPredict:    Prob 3
"""

import csv
import os
import pprint
import random
import time

import numpy as np
import torch
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
from PIL import Image, ImageEnhance, ImageFilter
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import reader
import utils

class TrimmedVideos(Dataset):
    def __init__(self, root, train: bool, downsample=1, rescale=1, sample=None, transform=None):
        if train:
            self.label_path = os.path.join(root, "label", "gt_train.csv")
            self.video_path = os.path.join(root, "video", "train")
        else:
            self.label_path = os.path.join(root, "label", "gt_valid.csv")
            self.video_path = os.path.join(root, "video", "valid")

        self.downsample = downsample
        self.rescale    = rescale
        self.sample     = sample
        self.transform  = transform
        
        self.video_list, self.len = reader.getVideoList(self.label_path)
    
    def __len__(self):
        return self.len

    def __getitem__(self, index):
        video_name     = self.video_list['Video_name'][index]
        video_category = self.video_list['Video_category'][index]
        video_label    = torch.LongTensor([self.video_list['Action_labels'][index]])

        # ---------------------------------------------------------------
        # Sample for HW4.1, pick the fixed number of frames 
        # Downsample for HW4.2, pick the frames with the downsampling rate
        # ----------------------------------------------------------------
        video = reader.readShortVideo(self.video_path, video_category, video_name, downsample_factor=self.downsample, rescale_factor=self.rescale)
        # print("Video.shape: {}".format(video.shape))
        # print("Video.type:  {}".format(type(video)))

        if self.sample:
            frame = np.arange(0, video.shape[0], (video.shape[0] // self.sample + 1))
            video = video[frame]

        if self.transform:
            tensor = torch.zeros(video.shape[0], 3, 240, 320).type(torch.float32)
            
            for i in range(video.shape[0]):
                tensor[i] = self.transform(video[i])

        else:
            tensor = torch.from_numpy(video).permute(0, 3, 1, 2).type(torch.float32) / 255.0
        
        return tensor, video_label

class FullLengthVideos(Dataset):
    def __init__(self, root, train: bool, downsample=1, rescale=1, length=32, transform=None):
        if train:
            self.label_path = os.path.join(root, "labels", "train")
            self.video_path = os.path.join(root, "videos", "train")
        else:
            self.label_path = os.path.join(root, "labels", "valid")
            self.video_path = os.path.join(root, "videos", "valid")

        self.train      = train
        self.length     = length
        self.downsample = downsample
        self.rescale    = rescale
        self.transform  = transform

        self.video_category = [label.split('.')[0] for label in os.listdir(self.label_path)]
        
        self.len = 0

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        pass

class TrimmedVideosPredict(Dataset):
    def __init__(self, video_folder, downsample=1, rescale=1, sample=None, transform=None):
        self.video_path = video_folder
        self.downsample = downsample
        self.rescale    = rescale
        self.sample     = sample
        self.transform  = transform
        
        self.video_list, self.len = reader.getVideoList(self.video_path)
    
    def __len__(self):
        return self.len

    def __getitem__(self, index):
        video_name     = self.video_list['Video_name'][index]
        video_category = self.video_list['Video_category'][index]
        video_label    = torch.LongTensor([self.video_list['Action_labels'][index]])

        # ---------------------------------------------------------------
        # Sample for HW4.1, pick the fixed number of frames 
        # Downsample for HW4.2, pick the frames with the downsampling rate
        # ----------------------------------------------------------------
        video = reader.readShortVideo(self.video_path, video_category, video_name, downsample_factor=self.downsample, rescale_factor=self.rescale)
        total_frame = video.shape[0]

        if self.sample:
            frame_to_catch = [int((i + 0.5) * (total_frame // self.sample)) for i in range(0, self.sample)]
        
        if self.transform:
            frames = torch.cat([self.transform(video[f]).unsqueeze(0) for f in frame_to_catch], dim=0)

        return frames, video_label

def video_unittest(data_path):
    dataset = TrimmedVideos(data_path, train=True, downsample=1, sample=4, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]))

    dataloader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=8)
    iterator = iter(dataloader)
    
    for _ in range(10):
        frames, labels = next(iterator)
        labels = labels.view(-1)
        print("Frames.shape: \t{}".format(frames.shape))
        # print(frames)
        print("Labels.shape: \t{}".format(labels.shape))
        print(labels)

    return

def predict_unittest(data_path):
    dataset = TrimmedVideosPredict(data_path, downsample=1, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]))

    dataloader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=8)
    iterator = iter(dataloader)
    
    return

def continuous_images_unittest(data_path):
    pass

def continuous_predict_unittest(data_path):
    pass

def main():
    datapath = "./hw4_data/TrimmedVideos/"

    video_unittest(datapath)
    print("Video Unittest Passed!")

    predict_unittest(os.path.join(datapath, "video", "valid"))
    print("Predict Unittest Passed!")

    # datapath = "./hw4_data/FullLengthVideos"
    
    # continuous_images_unittest(datapath)
    # print("Continuous Images Unittest Passed!")

    # continuous_predict_unittest(datapath)
    # print("Continuous Predict Unittest Passed!")

if __name__ == "__main__":
    main()
