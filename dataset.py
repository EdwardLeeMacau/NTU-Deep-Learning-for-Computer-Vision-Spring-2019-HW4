"""
  FileName     [ dataset.py ]
  PackageName  [ HW4 ]
  Synopsis     [ Dataset of the HW4 ]
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

import reader
import utils


class TrimmedVideos(Dataset):
    def __init__(self, root, train: bool, downsample=None, rescale=1, sample=None, transform=None):
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
        total_frame = video.shape[0]

        if self.sample:
            frame_to_catch = [int((i + 0.5) * (total_frame // self.sample)) for i in range(0, self.sample)]
        
        if self.transform:
            frames = torch.cat([self.transform(video[f]).unsqueeze(0) for f in frame_to_catch], dim=0)

        return frames, video_label

class FullLengthVideos(Dataset):
    def __init__(self, root, transform=None):
        pass

    def __len__(self):
        pass

    def __getitem__(self, index):
        pass

class TrimmedVideosPredict(Dataset):
    def __init__(self, video_folder, downsample=1, rescale=1, transform=None):
        self.video_path = video_folder
        self.transform  = transform
        
    def __len__(self):
        pass

    def __getitem__(self, index):
        pass

def video_unittest(data_path):
    dataset = TrimmedVideos(data_path, train=True, downsample=1, sample=4, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]))

    dataloader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=8)
    iterator = iter(dataloader)
    
    for i in range(10):
        frames, labels = next(iterator)
        labels = labels.view(-1)
        print("Frames.shape: \t{}".format(frames.shape))
        # print(frames)
        print("Labels.shape: \t{}".format(labels.shape))
        print(labels)

    return

def predict_unittest(data_path):
    return

def continuous_images_unittest(data_path):
    pass

def continuous_predict_unittest(data_path):
    pass

def main():
    datapath = "./hw4_data/TrimmedVideos/"

    video_unittest(datapath)
    print("Video Unittest Passed!")

    # predict_unittest(datapath)
    # print("Predict Unittest Passed!")

    # datapath = "./hw4_data/FullLengthVideos"
    
    # continuous_images_unittest(datapath)
    # print("Continuous Images Unittest Passed!")

    # continuous_predict_unittest(datapath)
    # print("Continuous Predict Unittest Passed!")

if __name__ == "__main__":
    main()
