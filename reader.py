"""
  FileName     [ reader.py ]
  PackageName  [ HW4 ]
  Synopsis     [ Sample code to load an video ]

  Library:
    scikit-video    1.1.11
    numpy           1.16.2
    ffmpeg
    ffprobe
"""

import numpy as np
import skvideo.io
import skimage.transform
import csv
import collections
import os
import pprint

def readShortVideo(video_path, video_category, video_name, downsample_factor=12, rescale_factor=1):
    '''
    @param video_path: video directory
    @param video_category: video category (see csv files)
    @param video_name: video name (unique, see csv files)
    @param downsample_factor: number of frames between each sampled frame (e.g., downsample_factor = 12 equals 2fps)
    @param rescale_factor: float of scale factor (rescale the image if you want to reduce computations)

    @return: (T, H, W, 3) ndarray, T indicates total sampled frames, H and W is heights and widths
    '''

    filepath = video_path + '/' + video_category
    filename = [file for file in os.listdir(filepath) if file.startswith(video_name)]
    video = os.path.join(filepath,filename[0])

    videogen = skvideo.io.vreader(video)
    frames = []
    for frameIdx, frame in enumerate(videogen):
        if frameIdx % downsample_factor == 0:
            frame = skimage.transform.rescale(frame, rescale_factor, mode='constant', preserve_range=True, multichannel=True, anti_aliasing=True).astype(np.uint8)
            frames.append(frame)
        else:
            continue

    return np.array(frames).astype(np.uint8)


def getVideoList(data_path):
    '''
    @param data_path: ground-truth file path (csv files)

    @return: ordered dictionary of videos and labels {'Action_labels', 'Nouns', 'End_times', 'Start_times', 'Video_category', 'Video_index', 'Video_name'}
    '''
    result    = {}
    num_video = 0

    with open(data_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            num_video += 1
            for column, value in row.items():
                result.setdefault(column,[]).append(value)

    od = collections.OrderedDict(sorted(result.items()))
    return od, num_video

def unittest():
    # print(ffmpeg.__file__)
    # skvideo.setFFmpegPath("C://Users//Edward Lee//AppData//Local//Programs//Python//Python37//lib//site-packages//ffmpeg")

    data_path  = "./hw4_data/TrimmedVideos/label/gt_train.csv"
    video_list, num_video = getVideoList(data_path)
    video_path = "./hw4_data/TrimmedVideos/video/train"

    for i in range(num_video):
        video_name     = video_list['Video_name'][i]
        video_category = video_list['Video_category'][i]

        video = readShortVideo(video_path, video_category, video_name, downsample_factor=1, rescale_factor=1)
        print(i, video.shape)

if __name__ == "__main__":
    unittest()
    print("Unittest Passed!")
