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

# import matplotlib as plt

import os

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from SizeEstimator import SizeEstimator

import nibabel as nib
from timeit import default_timer as timer

from dataloader_dev import RSOMLayerDataset, RandomZShift, ZeroCenter, CropToEven, ToTensor
    

# root_dir = '/home/gerlstefan/data/dataloader_dev'
root_dir = '/home/gerlstefan/data/fullDataset/labeled'

os.environ["CUDA_VISIBLE_DEVICES"]='0'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')
model = UNet(in_channels=3,
             n_classes=2,
             depth=3,
             wf=6,
             padding=True,
             batch_norm=True,
             up_mode='upsample').to(device)

model = model.float()

print('Current GPU device:', torch.cuda.current_device())
print('model down_path first weight at', model.down_path[0].block.state_dict()['0.weight'].device)

optim = torch.optim.Adam(model.parameters(), lr=1e-2)

Obj = RSOMLayerDataset(root_dir, transform=transforms.Compose([RandomZShift(), ZeroCenter(), CropToEven(), ToTensor()]))
dataloader = DataLoader(Obj, batch_size=1, shuffle=False, num_workers=4, pin_memory = True)

epochs = 10
train = True

# TODO: custom minibatch size 
# TODO: add mirror padding beforehand??
# TODO: torch data type
# TODO: understand output shape?

GPU_RAM = 16000
est_ram = 0
# minibatch_size = 1
# with torch.no_grad():
#     while est_ram < 0.9*GPU_RAM:
#         se = SizeEstimator(model, input_size=(minibatch_size,
#             Obj[0]['data'].shape[1],
#             Obj[0]['data'].shape[2], 
            # Obj[0]['data'].shape[3]))
        # est_ram = se.estimate_size()

# print(minibatch_size)
minibatch_size = 10 
# start timing
timing = 0

if timing:
    start = timer()
    start_global = start

for i in range(epochs):
    print('Epoch:',i+1, 'of', epochs)
    for i_batch, batch in enumerate(dataloader):
        if timing:
            start_batch = timer()
        print('   Batch:', i_batch+1, 'of', len(Obj)) 
        # if i_batch>=1:
        #     print('Loop breaking')
        #     break
        # print(batch['data'].device)
        batch['label'] = batch['label'].to(device)
        batch['data'] = batch['data'].to(device)
        # print(batch['data'].device)
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
            if train:
                if timing:
                    stop = timer()
                    print('Everything else', stop - start)
                # start = timer()
                # X = X.to(device)  # [N, 3, H, W] # replace 3 with 2 channels
                # Y = Y.to(device)  # [N, H, W] with class indices (0, 1)
                # stop = timer()
                # print('to device', stop - start)
                    start = timer()
                prediction = model(X)  # [N, 2, H, W]
            
                #prediction = torch.squeeze(prediction)
                # if not i:
                #    print('prediction shape', prediction.shape)
                #    print('label shape', y.shape)
                    
                loss = F.cross_entropy(prediction, Y)
                
    
                optim.zero_grad()
                loss.backward()
                optim.step()
                if timing:
                    stop = timer()
                    print('Pred and loss', stop - start)
                    start = timer()
        if timing:
            stop_batch= timer()
            msg  = 'Batch:', stop_batch - start_batch
            sys.stdout.write("\r" + msg)
            sys.stdout.flush()
            # print(torch.cuda.max_memory_allocated()*1e-6,'MB memory used')

    if train:            
        print('Loss:', loss.item())
if timing:
    stop_global = timer()
    print('overall time', stop_global - start_global)
print(torch.cuda.max_memory_allocated()*1e-6,'MB memory used')





def train(model, iterator, optimizer, args):
    '''
    train one epoch
    '''
    # get the next batch of training data
    for i in range(args.size_train)
        batch = next(iterator)

    








