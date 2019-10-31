#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 21:31:01 2019

@author: stefan
"""

import numpy as np
from scipy import ndimage
import nibabel as nib

# try and generate artificial RSSOM noise
# first try:
# try to generate noise "in between" vessels

# with convolution of shell


# generate filter for convolution
struct = ndimage.generate_binary_structure(3, 1)

struct = ndimage.iterate_structure(struct, 7).astype(int)

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
    

# load vessel file
file = '/home/stefan/PYTHON/HQDatasetVesselAnnot/vessels/R_20190605163439_HQ0003_th_corrected.nii.gz'
#file = '/home/stefan/PYTHON/HQDatasetVesselAnnot/test_noise_generation/1.nii.gz'

file_handle = nib.load(file)
label = file_handle.get_fdata()
label = label.astype(int)

A = ndimage.convolve(label, shell)

A = np.logical_and(A>=1, A<=3)


# generate filter for convolution
struct = ndimage.generate_binary_structure(3, 1)

struct = ndimage.iterate_structure(struct, 15).astype(int)

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
            


B = ndimage.convolve(label, shell)

B = np.logical_and(B>=1, B<=3)

C = np.logical_or(A, B)

# missing add irregularity plus maybe another line in between
# maybe lines thicker? like 2 pixels
C = ndimage.morphology.binary_dilation(C)


outfile = '/home/stefan/PYTHON/HQDatasetVesselAnnot/test_noise_generation/R_20190605163439_HQ0003_noise_2_dil.nii.gz'

#outfile = '/home/stefan/PYTHON/HQDatasetVesselAnnot/test_noise_generation/1_noise.nii.gz'

img = nib.Nifti1Image(C.astype(np.uint8), np.eye(4))
nib.save(img, outfile)








        



