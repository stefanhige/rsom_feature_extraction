#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 16:36:48 2020

@author: stefan
"""
import torch
import numpy as np
import os
import nibabel as nib

def _dice(x, y):
    '''
    do the test in numpy
    '''
    if isinstance(x, torch.Tensor):
        x = x.cpu().numpy()
    if isinstance(y, torch.Tensor):
        y = y.cpu().numpy()

    x = x.astype(np.bool)
    y = y.astype(np.bool)

    i = np.logical_and(x,y)

    if x.sum() + y.sum() == 0:
        return 1.

    return (2. * i.sum()) / (x.sum() + y.sum())


def load_seg(path):
    img = nib.load(path)
    return img.get_data()

def dice_nii(x, y):
    x_ = load_seg(x)
    y_ = load_seg(y)
    print(os.path.basename(x), _dice(x_,y_))
    
    



pred_dir = '/home/stefan/fbserver_ssh/data/layerunet/miccai/dataset-cleanup/eval'
gt_dir = '/home/stefan/fbserver_ssh/data/layerunet/fullDataset/labeled/val'


files = os.listdir(pred_dir)

files = [el for el in files if not 'ppred' in el]


for el in files:
    dice_nii(os.path.join(pred_dir,el),
             os.path.join(gt_dir, el.replace('pred','l')))


