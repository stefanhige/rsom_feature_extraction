#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 15:38:26 2019

@author: stefan
"""
import nibabel as nib
import numpy as np
import imageio
import os
from scipy import ndimage


def load_seg(path):
    img = nib.load(path)
    return img.get_data()
    
def load_rgb(path):
    img = nib.load(path)
    data = img.get_data()
    data = np.stack([data['R'], data['G'], data['B']], axis=-1)
    return data


def cut(vol_file, label_file, destination):
    
    vol = load_rgb(vol_file)
    
    label = load_seg(label_file)
    
    print(vol.shape, label.shape)
    
    label = label[:vol.shape[0],...]
    
    print(vol.shape, label.shape)

    img = nib.Nifti1Image(label, np.eye(4))   
    path = os.path.join(destination,os.path.basename(label_file))
    nib.save(img, path)
    
    
    
origin = '/home/stefan/data/vesnet/cutDataset'
destination = '/home/stefan/data/vesnet/cutDataset/res'
    
cwd = os.getcwd()
# change directory to origin, and get a list of all files
os.chdir(origin)
all_files = os.listdir()
os.chdir(cwd)
filenames = [el for el in all_files if '_v_rgb.nii.gz' in el]

#filenames = [filenames[0]]

for file in filenames:
    
    label_file = file.replace('_v_rgb.nii.gz','_v_l.nii.gz')
    file = os.path.join(origin, file)
    label_file = os.path.join(origin, label_file)
    
    cut(file, label_file, destination)
    
    