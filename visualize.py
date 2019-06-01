"""
  FileName     [ visualize.py ]
  PackageName  [ HW4 ]
  Synopsis     [ Seq-Seq result visualization ]
"""

import argparse
import datetime
import itertools
import os
import pprint
import random
from datetime import date

import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt
from PIL import Image
from torch import nn
from torchvision import transforms
from torchvision.utils import make_grid
from tqdm import tqdm

import dataset
import utils
import visualize
from classifier import Classifier
from cnn import resnet50
from rnn import LSTM_Net

# Set as true when the I/O shape of the model is fixed

parser = argparse.ArgumentParser()
# Basic setting
parser.add_argument("--sample", default=5, type=int, help="the number of sample images of the video.")
parser.add_argument("--kernel", default=3, type=int, help="the number of sample images of the video.")
# Message logging, model saving setting
parser.add_argument("--tag", default="20190529_2", type=str, help="tag for this training")
# Load dataset, pretrain model setting
parser.add_argument("--video", default="./hw4_data/FullLengthVideos/videos/train", type=str, help="path to load train datasets")
parser.add_argument("--label", default="./hw4_data/FullLengthVideos/labels/train", type=str, help="path to load validation datasets")

opt = parser.parse_args()

def post_process(label):
    """
      Using simple method to smooth the label, enhance the seq-to-seq prediction accuracy.

      Params:
      - label: The action label

      Return:
      - label: The action label after smoothing 
    """
    # -------------------------
    # dimension:
    #   label: (batch, frames)
    # -------------------------
    frames = len(label)
    return label

def visualization(save_path, img_path, predict: np.array, label: np.array, sample=5, bar_height=20):
    """
      Visualize the video prediction performance with the timeline.

      Params:
      - save_path: the directory to save the generated images
      - img_path: the folder of the FullLengthVideos
      - predict: the predict actions
      - label: the ground truth actions

      Return: None
    """
    
    # -----------------------------------
    # Load and smaple frames in the video
    # -----------------------------------
    imgs = [os.path.join(img_path, name) for name in os.listdir(img_path)]
    step = (len(imgs) - 1) / sample
    keep = np.arange(0, len(imgs), step).astype(int)[:5].tolist()
    
    images = []
    for k in keep:
        img = transforms.ToTensor()(Image.open(imgs[k]))
        images.append(img)

    fig = np.array(transforms.ToPILImage()(make_grid(images)))

    # -------------------------------------
    # Make the color bar to show the labels
    # -------------------------------------
    h, w, c  = fig.shape
    bar_gt   = convert_bar(label, w, bar_height=bar_height)
    bar_pred = convert_bar(predict, w, bar_height=bar_height)
    fig      = np.concatenate((bar_gt, fig, bar_pred), axis=0)

    plt.imsave(save_path, fig)
    
    return

def convert_bar(label, bar_width, bar_height=20, color=plt.get_cmap('Set1')) -> np.ndarray:
    """ Make the label colorbar """
    mark = convert_marks(label)

    rects = []
    for start, length, k in mark:
        lt, rb = rectangle(start, start + length, label.shape[0], bar_width, bar_height)
        rects.append((lt, rb, k))
    
    # Shape of image in numpy is (height, width, channel)
    bar = np.zeros((bar_height, bar_width, 3), np.uint8)
    for lt, rb, k in rects:
        r, g, b, a = (c * 255 for c in color(k))
        bar = cv2.rectangle(bar, lt, rb, (r, g, b, a), thickness=-1)

    return bar

def convert_marks(label):
    """ Convert the per frames label into marks (startpoint, length, label)"""
    marks = []
    start = 0
    for k, g in itertools.groupby(label):
        length = len(list(g))
        marks.append((start, length, k))
        start += length
        
    return marks

def rectangle(start, end, length, max_width, height):
    """ Return the coordinate of (left, top) and (right, bottom) """
    print(start, end, length, max_width, height)
    left  = int(start / length * max_width)
    right = int(end / length * max_width)
    
    return (left, 0), (right, height)

def post_process_unittest():
    label = np.random.rand(1, 768)
    post_process(label)
    print(label)

    return

def visualization_unittest():
    root_path  = os.path.join("hw4_data", "FullLengthVideos", "videos", "train")
    category   = os.listdir(root_path)
    label_path = os.path.join("hw4_data", "FullLengthVideos", "labels", "train")

    video_path = os.path.join(root_path, category[0])
    label = np.loadtxt(os.path.join(label_path, category[0] + '.txt'), dtype=int)
    visualization('visualization.png', video_path, label, label, sample=5)

    return

def main():
    visualization_unittest()
    print("Visualization Unittest Passed!")

    return

if __name__ == "__main__":
    main()
