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


def reflection_noise():



# load input file

origin = '/home/stefan/data/vesnet/synthDataset/rsom_style_noisy_small/train/1_v_rgb.nii.gz'

data = (nib.load(origin)).get_data()
data = np.stack([data['R'], data['G'], data['B']], axis=-1)
# data = data.astype(np.uint8)


noise_red = np.zeros_like(data[...,0])


#choose a dimension
dim = np.random.randint(3)

# choose an index in that dimension (not so close to the boundaries)

# probably redundant, only true if shape[dim] == 0
assert int(0.9*noise_red.shape[dim])+1 <= noise_red.shape[dim]

idx_along_dim = np.random.randint(int(0.1*noise_red.shape[dim]),
                                  int(0.9*noise_red.shape[dim])+1)


print(dim, idx_along_dim)

# area of that plane with normal dimension
print(noise_red.shape[dim-1], noise_red.shape[dim-2])
area = noise_red.shape[dim-1] * noise_red.shape[dim-2]
print(area)

for it in range(int(area/1000)):
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
red_channel[red_channel<50] = 0
red_channel = red_channel.astype(np.bool)

noise_red = noise_red * np.logical_not(red_channel)

noise_red[noise_red < 0] = 0
data[...,0] += 3*noise_red.astype(np.uint8)

shape_3d = data.shape[0:3]
rgb_dtype = np.dtype([('R', 'u1'), ('G', 'u1'), ('B', 'u1')])
#Vm_nii = Vm.astype('u1')
Vm_nii = data.copy().view(rgb_dtype).reshape(shape_3d)
img = nib.Nifti1Image(Vm_nii, np.eye(4))
nib.save(img, 'refl_noise_rgb.nii.gz')

#save segmentation
refl_noise = nib.Nifti1Image(noise_red.astype(np.uint8), np.eye(4))
nib.save(refl_noise, 'refl_noise.nii.gz')
