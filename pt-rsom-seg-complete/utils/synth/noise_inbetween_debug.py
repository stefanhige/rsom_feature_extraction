#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  6 21:07:41 2019

@author: stefan
"""

# thinning of vessels up to a certain cutoff
import nibabel as nib
import numpy as np

from scipy import ndimage
from skimage import morphology
from skimage import exposure

def noise_type_inbetween(inputVolume):
    
    inputVolume = inputVolume.astype(np.uint8)
    
    # generate filter for convolution 
    struct = morphology.ball(7)
    l = struct.shape[0]
    shell = np.zeros(struct.shape, dtype=int)
    
    for x in np.arange(l):
        for y in np.arange(l):
            nz = np.nonzero(struct[x,y,:])[0]
            if len(nz) == 1:
                shell[x,y,nz[0]] = 1
            elif len(nz) >= 2:
                shell[x,y,nz[0]] = 1
                shell[x,y,nz[-1]] = 1
                
        
    
    A = ndimage.convolve(inputVolume, shell)
    A = np.logical_and(A>=1, A<=3)

    # generate filter for convolution
    struct = ndimage.generate_binary_structure(3, 1)
    
    struct = ndimage.iterate_structure(struct, 15).astype(int)
    
    
    struct = morphology.ball(15)
    l = struct.shape[0]
    shell = np.zeros(struct.shape, dtype=int)
    
    for x in np.arange(l):
        for y in np.arange(l):
            nz = np.nonzero(struct[x,y,:])[0]
            if len(nz) == 1:
                shell[x,y,nz[0]] = 1
            elif len(nz) >= 2:
                shell[x,y,nz[0]] = 1
                shell[x,y,nz[-1]] = 1
                

    
    B = ndimage.convolve(inputVolume, shell)
    
    B = np.logical_and(B>=1, B<=3)
    
    V = np.logical_or(A, B)
    
    
    # random zero out entries
    mask = np.random.random_sample(V.shape)
    
    mask = mask >= 0.9
    
    V = mask * V
    
    V_mid = ndimage.binary_dilation(V)
    V_out = ndimage.binary_dilation(V_mid)
    
    V = V.astype(np.uint8) + V_mid.astype(np.uint8) + V_out.astype(np.uint8)
    
    # add another mask?
    mask2 = np.random.random_sample(V.shape)
    
    mask2 = mask2 >= 0.5
    
    V = mask2* V
    
    
    return V


file = '/home/stefan/PYTHON/HQDatasetVesselAnnot/test_noise_generation/1_RGB3_inside_larger_intensity_mod_l.nii.gz'


file_handle = nib.load(file)
SEG = file_handle.get_fdata()
SEG = SEG.astype(bool)


# add noise 
# "inbetween" noise

noise = noise_type_inbetween(SEG) 




# save noise, debug only
file_noise = file.replace('_l.nii.gz','')
file_noise = file_noise + '_noise_ball_mask.nii.gz'
        
nib.save(nib.Nifti1Image(noise.astype(np.uint8), np.eye(4)), file_noise)







