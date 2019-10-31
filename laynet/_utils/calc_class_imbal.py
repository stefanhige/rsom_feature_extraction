# calculate class imbalance
import torch

from torch import nn

import torch.nn.functional as F

import numpy as np

import os
import sys

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

from dataloader_dev import RSOMLayerDataset
from dataloader_dev import RandomZShift, ZeroCenter, CropToEven
from dataloader_dev import DropBlue, ToTensor, precalcLossWeight






# TODO:
# import training dataset
# loop over dataset
# extract label and count ones and zeros
# class0 = n_0/(n_0+n_1)
# class1 = n_1/(n_0+n_1)


dataset_dir = '/home/gerlstefan/data/fullDataset/labeled/train'

dataset = RSOMLayerDataset(dataset_dir,
        transform=transforms.Compose([RandomZShift(), ZeroCenter(), DropBlue(), ToTensor()]))


print('Length of dataset', len(dataset))
ones_sum = 0
zeros_sum = 0

for i in range(len(dataset)):
    label = dataset[i]['label']
    label = label.numpy()

    ones = np.sum(label)
    ones_sum += ones
    zeros = np.sum(np.logical_not(label))
    zeros_sum += zeros 

    print('Sample:', i+1)
    print('    Class 0:', zeros/label.size)
    print('    Class 1:', ones/label.size)
    assert ones+zeros == label.size

ones_mean = ones_sum/(label.size*len(dataset))
zeros_mean = zeros_sum/(label.size*len(dataset))

print('mean values:')
print('    Class 0:', zeros_mean)
print('    Class 1:', ones_mean)
