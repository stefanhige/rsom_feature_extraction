#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  9 17:49:58 2019

@author: sgerl
"""

import torch
import torch.nn.functional as F
from unet import UNet

import numpy as np

import matplotlib as plt

import os

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

import nibabel as nib

from dataloader_dev import RSOMLayerDataset, RandomZShift, ZeroCenter, CropToEven, ToTensor



def plotMIP():
    '''
    
    '''
    
    
def plotMIP_sliced():
    '''
    
    '''
    




root_dir = '/home/sgerl/Documents/PYTHON/TestDataset20190411/selection/layerseg/dataloader_dev'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = UNet(in_channels=3,
             n_classes=2,
             depth=3,
             wf=3,
             padding=True,
             batch_norm=True,
             up_mode='upsample').to(device)

model = model.float()


optim = torch.optim.Adam(model.parameters(), lr=1e-2)

Obj = RSOMLayerDataset(root_dir, transform=transforms.Compose([RandomZShift(), ZeroCenter(), CropToEven(), ToTensor()]))
dataloader = DataLoader(Obj, batch_size=1, shuffle=False, num_workers=0)

epochs = 1

# TODO: custom minibatch size 



# TODO: add mirror padding beforehand??
# TODO: torch data type
# TODO: understand output shape?
minibatch_size = 10

for i in range(epochs):
    print('Epoch:',i+1, 'of', epochs)
    for batch in dataloader:
        for it in np.arange(batch['data'].shape[1], step=minibatch_size):
            if it + minibatch_size < batch['data'].shape[1]:
                X = batch['data'][:, it:it+minibatch_size, :, :]
            else:
                X = batch['data'][:, it:, :, :]
            
            #print('Shape of X:', X.shape)
            
            X = torch.squeeze(X, dim=0)
            X = X.float() # float is 32bit floating point
            
            Y = batch['label'][:, it:it+minibatch_size, :, :]
            Y = torch.squeeze(Y, dim=0)
            Y = Y.long() # long is int64
#            if not i:
#                print('data shape', X.shape)
#                print('label shape', y.shape)
            if 1:
                X = X.to(device)  # [N, 3, H, W] # replace 3 with 2 channels
                Y = Y.to(device)  # [N, H, W] with class indices (0, 1)
                prediction = model(X)  # [N, 2, H, W]
            
                #prediction = torch.squeeze(prediction)
                #if not i:
                #    print('prediction shape', prediction.shape)
                #    print('label shape', y.shape)
                    
                loss = F.cross_entropy(prediction, Y)
                print('Loss:', loss.item())
    
                optim.zero_grad()
                loss.backward()
                optim.step()
                
