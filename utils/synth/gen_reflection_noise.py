#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  9 14:28:06 2019

@author: stefan
"""
import nibabel as nib
import numpy as np

from scipy import ndimage
from skimage import morphology

import torch
import os


def reflection_noise_1(data, noise_red, dim, idx_along_dim):
    
    # area of that plane with normal dimension
    print(noise_red.shape[dim-1], noise_red.shape[dim-2])
    area = noise_red.shape[dim-1] * noise_red.shape[dim-2]
    print(area)
    
    for it in range(int(np.random.randint(1,3)*(area/2000))):
        # so we have the index along the normal dimension,
        # generate the indices along the remaining dimension 
        idx1 = np.random.randint(int(0.1*noise_red.shape[dim-1]),
                                 int(0.9*noise_red.shape[dim-1])+1)
        
        idx2 = np.random.randint(int(0.1*noise_red.shape[dim-2]),
                                 int(0.9*noise_red.shape[dim-2])+1)
        
        
        coords = tuple(np.roll(np.array([idx_along_dim, idx2, idx1]), dim))
        
        try: 
            noise_red[coords]
            
        except:
            print('fail')
        
        #print(coords)
        #assign random points in the surface
        
        #TODO: draw values from a gaussian
        noise_red[coords] = 1
    
    
    # put a lot of padding!
    noise_red = np.pad(noise_red, 25, mode='edge')
    
    
    # generate filter for convolution
    struct = morphology.ball(31)
    struct = ndimage.zoom(struct, np.roll([0.28, 1, 1], dim))
    # do a 3d convolution to spread out the noise
    noise_red = torch.from_numpy(noise_red)
    struct = torch.from_numpy(struct)
    noise_red = torch.unsqueeze(torch.unsqueeze(noise_red, 0), 0)
    struct = torch.unsqueeze(torch.unsqueeze(struct, 0), 0)
    
    noise_red = torch.nn.functional.conv3d(noise_red.float().cuda(), 
                                           struct.float().cuda())
    
    
    struct = morphology.ball(3)
    struct = struct/343
    struct = torch.from_numpy(struct)
    struct = torch.unsqueeze(torch.unsqueeze(struct, 0), 0)
    
    noise_red = torch.nn.functional.conv3d(noise_red, struct.float().cuda())
    
    
    # convolve by another smaller ball
    
    #noise_red = ndimage.convolve(noise_red, struct)
    
    noise_red = noise_red.squeeze().cpu().numpy()
    
    #noise_red[noise_red >=2] -= 1
    print(np.amax(noise_red))
    noise_red = 4*noise_red
    noise_red[noise_red < 1] = 1
    
    noise_red -= 1
    noise_red[noise_red<0] = 0
    noise_red[noise_red >4 ] = 4
    
    noise_red = np.round(noise_red/noise_red.max() * 10)*3
    
    gauss = np.random.normal(scale=5, size=noise_red.shape)
    gauss = gauss * noise_red.astype(np.bool)
    
    noise_red += gauss
    noise_red[noise_red<0] = 0
    
    noise_red = ndimage.gaussian_filter(noise_red, sigma=3)
    
    gauss = np.random.normal(scale=3, size=noise_red.shape)
    gauss = gauss * noise_red.astype(np.bool)
    noise_red += gauss
    noise_red[noise_red<0] = 0
    
    noise_red = ndimage.gaussian_filter(noise_red, sigma=1)
    
    
    # align padding
    # torch can do that better, cause accept negative values
    
    noise_red = torch.from_numpy(noise_red)
    
    offs = np.asarray(data.shape[:-1]) - np.asarray(noise_red.shape)
    
    pad = (int(np.floor(offs[-1]/2)), int(np.ceil(offs[-1]/2)),
           int(np.floor(offs[-2]/2)), int(np.ceil(offs[-2]/2)),
           int(np.floor(offs[-3]/2)), int(np.ceil(offs[-3]/2)))
    
    noise_red = torch.nn.functional.pad(noise_red, pad)
    
    noise_red = noise_red.numpy()
    
    red_channel = data[...,0].copy()
    red_channel[red_channel<90] = 0
    red_channel = red_channel.astype(np.bool)
    
    noise_red = noise_red * np.logical_not(red_channel)
    
    noise_red[noise_red < 0] = 0
    
    return noise_red

def reflection_noise(data):
    noise_red = np.zeros_like(data[...,0])
    #choose a dimension
    dim = np.random.randint(3)
    
    # choose an index in that dimension (not so close to the boundaries)
    # probably redundant, only true if shape[dim] == 0
    assert int(0.9*noise_red.shape[dim])+1 <= noise_red.shape[dim]
    
    idx_along_dim = np.random.randint(int(0.1*noise_red.shape[dim]),
                                      int(0.9*noise_red.shape[dim])+1)
    
    
    print(dim, idx_along_dim)
    
    #if np.random.randint(0,4) >= 3:
    if 1:   
        noise_red1 = reflection_noise_1(data, noise_red.copy(), dim, idx_along_dim)
        
        # generate a second index close to the first one
        
        idx_offs = np.random.randint(20, 50)
        print('offset for second layer is:', idx_offs)
        if idx_along_dim > data.shape[dim]/2:
            idx_offs = -idx_offs
        
        new_idx = idx_along_dim + idx_offs
        
        if new_idx < 10 or new_idx >= data.shape[dim] - 10:
            print('one noise layer is enough')
        else:
            if np.random.randint(0, 2):
                # TODO
                # more random
                print('add second layer')
                noise_red2 = reflection_noise_1(data, noise_red.copy(), dim, new_idx)
                data[...,0] += np.random.randint(2,4)*noise_red2.astype(np.uint8)
            else:
                print('only one layer!')
        
        # multiply by 2 or 3
        data[...,0] += np.random.randint(2,4)*noise_red1.astype(np.uint8)
        return data


root_dir = '/home/stefan/data/vesnet/synthDataset/rsom_style_noisy/train'
dest_dir = '/home/stefan/data/vesnet/synthDataset/rsom_style_noisy+refl/train'

# change directory to origin, and get a list of all files
all_files = os.listdir(root_dir)
all_files.sort()

# extract the n.nii.gz files
filenames = [el for el in all_files if el[-10:] == 'rgb.nii.gz']

#filenames = [filenames[0]]

for filename in filenames:
    
    print('file:', filename)
    origin = os.path.join(root_dir, filename)
    dest = os.path.join(dest_dir, filename)
    
    data = (nib.load(origin)).get_data()
    data = np.stack([data['R'], data['G'], data['B']], axis=-1)

    # 50% of the data has reflection
    if np.random.sample() > 0.5:
        print('add noise')
        data = reflection_noise(data)
    else:
        print('leave this one untouched')
        
    shape_3d = data.shape[0:3]
    rgb_dtype = np.dtype([('R', 'u1'), ('G', 'u1'), ('B', 'u1')])
    #Vm_nii = Vm.astype('u1')
    Vm_nii = data.copy().view(rgb_dtype).reshape(shape_3d)
    img = nib.Nifti1Image(Vm_nii, np.eye(4))
    nib.save(img, dest)


