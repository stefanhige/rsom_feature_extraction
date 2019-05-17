#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 10 14:48:49 2019

dataloader test script

@author: sgerl
"""

import torch
import numpy as np
import matplotlib as plt

import os

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

import nibabel as nib


class RSOMLayerDataset(Dataset):
    """rsom dataset class for layer segmentation"""

    def __init__(self, 
                 root_dir, 
                 data_str='_rgb.nii.gz', 
                 label_str='_l.nii.gz', 
                 transform=None):
        """
        Args:
            root_dir (string): Directory with all the nii.gz files.
            data_str (string): end part of filename of training data.
            label_str (string): end part of filename of segmentation ground truth data.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        assert os.path.exists(root_dir) and os.path.isdir(root_dir), \
        'root_dir not a valid directory'
        
        self.root_dir = root_dir
        self.transform = transform
        
        assert isinstance(data_str, str) and isinstance(label_str, str), \
        'data_str or label_str not valid.'
        
        self.data_str = data_str
        self.label_str = label_str
        
        # get all files in root_dir
        all_files = os.listdir(path = root_dir)
        # extract the  data files
        self.data = [el for el in all_files if el[-len(data_str):] == data_str]
        
        assert len(self.data) == \
            len([el for el in all_files if el[-len(label_str):] == label_str]), \
            'Amount of data and label files not equal.'

    def __len__(self):
        return len(self.data)
    
    @staticmethod
    def _readNII(rpath):
        '''
        read in the .nii.gz file
        Args:
            rpath (string)
        '''
        
        img = nib.load(str(rpath))
        
        # TODO: when does nib get_fdata() support rgb?
        # currently not, need to use old method get_data()
        return img.get_data()

    def __getitem__(self, idx):
        data_path = os.path.join(self.root_dir, 
                            self.data[idx])
        label_path = os.path.join(self.root_dir, 
                                   self.data[idx].replace(self.data_str, self.label_str))
        
        # read data
        data = self._readNII(data_path)
        data = np.stack([data['R'], data['G'], data['B']], axis=-1)
        data = data.astype(np.float64)
        
        # read label
        label = self._readNII(label_path)
        label = label.astype(np.float64)
        
        sample = {'data': data, 'label': label}

        if self.transform:
            sample = self.transform(sample)

        return sample
    


# transform
class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        data, label = sample['data'], sample['label']
        
        # data is [Z x X x Y x 3] [500 x 171 x 333 x 3]
        # label is [Z x X x Y] [500 x 171 x 333]
        
        # we want one sample to be [Z x Y x 3]  2D rgb image
        
        # numpy array size of images
        # [H x W x C]
        # torch tensor size of images
        # [C x H x W]
        
        # and for batches
        # [B x C x H x W]
        
        # here, X is the batch size.
        # so we want to reshape to
        # [X x C x Z x Y] [171 x 3 x 500 x 333]
        data = data.transpose((1, 3, 0, 2))
        
        # and for the label
        # [X x Z x Y] [171 x 500 x 333]
        label = label.transpose((1, 0, 2))
        
        return {'data': torch.from_numpy(data),
                'label': torch.from_numpy(label)}
        

# class random z-shift      
class RandomZShift(object):
    """Apply random z-shift to sample.

    Args:
        max_shift (int, tuple of int):  maximum acceptable 
                                        shift in -z and +z direction (in voxel)
        
    """

    def __init__(self, max_shift=0):
        assert isinstance(max_shift, (int, tuple))
        if isinstance(max_shift, int):
            self.max_shift = (-max_shift, max_shift)
        else:
            assert len(max_shift) == 2
            assert max_shift[1] > max_shift[0]
            self.max_shift = max_shift

    def __call__(self, sample):
        data, label = sample['data'], sample['label']
        assert isinstance(data, np.ndarray)
        assert isinstance(label, np.ndarray)
        
        # initial shape
        data_ishape = data.shape
        label_ishape = label.shape
        
        # generate random dz offset
        dz = int(round((self.max_shift[1] - self.max_shift[0]) * np.random.random_sample() + self.max_shift[0]))
        assert (dz >= self.max_shift[0] and dz <= self.max_shift[1])
        
        if dz:
            shift_data = np.zeros(((abs(dz), ) + data.shape[1:]), dtype = np.uint8)
            shift_label = np.zeros(((abs(dz), ) + label.shape[1:]), dtype = np.uint8)
        
            print('RandomZShift: Check if this array modification does the correct thing before actually using it')
            # TODO: verify
            data = np.concatenate((data[:-abs(dz),:,:,:], shift_data) if dz > 0 else (shift_data, data[abs(dz):,:,:,:]), axis = 0)
            label = np.concatenate((label[:-abs(dz),:,:], shift_label) if dz > 0 else (shift_label, label[abs(dz):,:,:]), axis = 0)

            # should be the same...
            assert (data_ishape == data.shape and label_ishape == label.shape)
        
        return {'data': data, 'label': label}
    
# class normalize
class ZeroCenter(object):
    """ 
    Zero center input volumes
    """    
    def __call__(self, sample):
        data, label = sample['data'], sample['label']
        assert isinstance(data, np.ndarray)
        assert isinstance(label, np.ndarray)
        
        # compute for all x,y,z mean for every color channel
        rgb_mean = np.around(np.mean(data, axis=(0, 1, 2))).astype(np.int16)
        meanvec = np.tile(rgb_mean, (data.shape[:-1] + (1,)))
        
        data -= meanvec
        
        return {'data': data, 'label': label}
    
 
# TODO CLASS implementation
class CropToEven(object):
    """ 
    if Volume shape is not even numbers, simply crop the first element
    except for last dimension, this is RGB  = 3
    """
    def __call__(self, sample):
        data, label = sample['data'], sample['label']
        assert isinstance(data, np.ndarray)
        assert isinstance(label, np.ndarray)
     
        IsOdd = np.mod(data.shape[:-1], 2)
        
        data = data[IsOdd[0]:, IsOdd[1]:, IsOdd[2]:, : ]
        label = label[IsOdd[0]:, IsOdd[1]:, IsOdd[2]:]
    
            
        return {'data': data, 'label': label}
            
        
        
    
        
        
# TODO:
# normalization?
# zero centering?
# torch tensor image visualization+
# of mip

        


# ==============================TEST===========================================

#root_dir = '/home/sgerl/Documents/PYTHON/TestDataset20190411/selection/layerseg/dataloader_dev'

#Obj = RSOMLayerDataset(root_dir, transform=CropToEven())

#sample = Obj[0]


#Obj = RSOMLayerDataset(root_dir, transform=transforms.Compose([RandomZShift(), ZeroCenter(), ToTensor()]))
#dat = Obj.readNII('/home/sgerl/Documents/PYTHON/TestDataset20190411/selection/layerseg/dataloader_dev/R_20170724150057_PAT001_RL01_l.nii.gz')

#sample = Obj[0]

#dataloader = DataLoader(Obj, batch_size=1, shuffle=False, num_workers=0)

# get a sample
#sample_dl = next(iter(dataloader))

#for sample in dataloader:
    #print(sample['data'].shape)

# numpy array size of images
# [H x W x C]
# torch tensor size of images
# [C x H x W]

# and for batches
# [B x C x H x W]

#np4d = sample['data']
#shape_3d = np4d.shape[0:3]
#rgb_dtype = np.dtype([('R', 'u1'), ('G', 'u1'), ('B', 'u1')])
#np4d = np4d.copy().view(rgb_dtype).reshape(shape_3d)
#img = nib.Nifti1Image(np4d, np.eye(4))
#       
#nib.save(img, '/home/sgerl/Documents/PYTHON/TestDataset20190411/selection/layerseg/dataloader_dev/out.nii.gz')