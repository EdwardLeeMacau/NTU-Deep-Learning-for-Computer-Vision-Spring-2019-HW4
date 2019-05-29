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
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader, Dataset

import reader
import utils
from cnn import resnet50

class TrimmedVideos(Dataset):
    def __init__(self, root, train: bool, downsample=1, rescale=1, sample=None, feature=False, transform=None):
        if train:
            self.label_path   = os.path.join(root, "label", "gt_train.csv")
            self.video_path   = os.path.join(root, "video", "train")
            self.feature_path = os.path.join(root, "feature", "train")
        else:
            self.label_path   = os.path.join(root, "label", "gt_valid.csv")
            self.video_path   = os.path.join(root, "video", "valid")
            self.feature_path = os.path.join(root, "feature", "valid")

        self.downsample = downsample
        self.rescale    = rescale
        self.sample     = sample
        self.transform  = transform
        self.feature    = feature
        
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
        if self.feature:
            video = reader.readShortFeature(self.feature_path, video_category, video_name, downsample_factor=self.downsample)
        else:
            video = reader.readShortVideo(self.video_path, video_category, video_name, downsample_factor=self.downsample, rescale_factor=self.rescale)
        
        if self.sample:
            step  = (video.shape[0] // self.sample) + 1
            frame = np.arange(0, video.shape[0], step)
            video = video[frame]

        # ---------------------------------------------------
        # Features Output dimension: (frames, 2048)
        # ---------------------------------------------------
        if self.feature:
            if self.transform:
                tensor = self.transform(video)
            else:
                tensor = torch.from_numpy(video)
            
            return tensor.squeeze(0), video_label

        # ---------------------------------------------------
        # Full video Output dimension: (frames, channel, height, width)
        # ---------------------------------------------------
        if self.transform:
            tensor = torch.zeros(video.shape[0], 3, 240, 320).type(torch.float32)
            for i in range(video.shape[0]):
                tensor[i] = self.transform(video[i])
        else:
            tensor = torch.from_numpy(video).permute(0, 3, 1, 2).type(torch.float32) / 255.0
        
        return tensor, video_label

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

class FullLengthVideos(Dataset):
    def __init__(self, root, train: bool, downsample=1, rescale=1, transform=None):
        if train:
            self.label_path = os.path.join(root, "labels", "train")
            self.video_path = os.path.join(root, "videos", "train")
        else:
            self.label_path = os.path.join(root, "labels", "valid")
            self.video_path = os.path.join(root, "videos", "valid")

        self.train      = train
        self.downsample = downsample
        self.rescale    = rescale
        self.transform  = transform
        self.video_list = [folder for folder in os.listdir(self.video_path)]

    def __len__(self):
        return len(self.video_list)

    def __getitem__(self, index):
        video_category = self.video_list[index]
        video_label    = os.path.join(self.label_path, video_category + '.txt')
        frame_names    = sorted([name for name in os.listdir(os.path.join(self.video_path, video_category))])

        # -------------------------------------------------------------
        # Full video Output dimension: (frames, channel, height, width)
        # -------------------------------------------------------------
        num_frames  = len(frame_names)
        keep_frames = np.arange(0, num_frames, self.downsample)
        bias_frames = np.zeros_like(keep_frames)
        frame_names = frame_names[keep_frames + bias_frames]

        if self.transform:
            tensor = torch.zeros(num_frames, 3, 240, 320).type(torch.float32)
            for i in range(num_frames):
                tensor[i] = self.transform(Image.open(frame_names[self.downsample * i]))
        else:
            tensor = torch.from_numpy(video).permute(0, 3, 1, 2).type(torch.float32) / 255.0
        
        return tensor, video_label, video_category

def read_feature_unittest(data_path):
    """ Read the videos in .npy format """
    for train in (True, False):
        dataset = TrimmedVideos(data_path, train=train, feature=True, downsample=1, transform=transforms.Compose([
            transforms.ToTensor(),
        ]))

        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

        if train:   train_val = "train"
        else:       train_val = "valid"

        for index, (data, label, category, name) in enumerate(dataloader, 1):
            data = data.squeeze(0)
            print("{:4d} {:16s} {:2d} {}".format(
                index, str(list(data.shape)), label[0].item(), os.path.join(data_path, "feature", train_val, category[0], name[0] + ".npy")))

    return

def video_unittest(data_path):
    """ Read the videos in .mp4 format """
    for train in (True, False):
        dataset = TrimmedVideos(data_path, train=True, downsample=1, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]))

        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
        
        if train:   train_val = "train"
        else:       train_val = "valid"

        for index, (data, label, category, name) in enumerate(dataloader, 1):
            data = data.squeeze(0)
            print("{:4d} {:16d} {:2d} {}".format(
                index, data.shape, label[0], os.path.join(data_path, "feature", train_val, category[0], name[0] + ".npy")))
                
    return

def predict_unittest(data_path):
    """ Read the videos in .mp4 format, without the ground truth file """
    for train in (True, False):
        dataset = TrimmedVideosPredict(data_path, downsample=1, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]))

        dataloader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=8)
    
        if train:   train_val = "train"
        else:       train_val = "valid"

        for index, (data, category, name) in enumerate(dataloader, 1):
            data = data.squeeze(0)
            print("{:4d} {:16d} {}".format(
                index, data.shape, os.path.join(data_path, "feature", train_val, category[0], name[0] + ".npy")))

    return

def continuous_images_unittest(data_path):
    pass

def continuous_predict_unittest(data_path):
    pass

def main():
    datapath = "./hw4_data/TrimmedVideos/"

    # video_unittest(datapath)
    # print("Video Unittest Passed!")

    read_feature_unittest(datapath)
    print("Read Features Unittest Passed!")

    # predict_unittest(os.path.join(datapath, "video", "valid"))
    # print("Predict Unittest Passed!")

    # datapath = "./hw4_data/FullLengthVideos"
    
    # continuous_images_unittest(datapath)
    # print("Continuous Images Unittest Passed!")

    # continuous_predict_unittest(datapath)
    # print("Continuous Predict Unittest Passed!")

if __name__ == "__main__":
    main()
