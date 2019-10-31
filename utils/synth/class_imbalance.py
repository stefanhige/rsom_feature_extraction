#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 16:14:14 2019

@author: stefan
"""

import os
import nibabel as nib
import numpy as np

# calculate class imbalace of synthetic dataset

root_dir = '/home/stefan/PYTHON/synthDataset/rsom_style'
all_files = os.listdir(root_dir)


all_files.sort()

# extract the label .nii.gz files
filenames = [el for el in all_files if el[-9:] == '_l.nii.gz']

running_true = 0.0

for filename in filenames: 
    
    origin = os.path.join(root_dir, filename)

    print('Processing ', filename)    
    # load input file
    label = (nib.load(origin)).get_fdata()
    label = label.astype(bool)
    
    true = np.sum(label) / np.size(label)
    print('True/False =', true)
    running_true += true
    
mean_true = running_true/len(filenames)

print('mean:', mean_true)

# mean is approx 0.014

# -> put 74.5 times positive weight on loss


