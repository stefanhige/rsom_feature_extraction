#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  6 22:29:08 2019

@author: stefan
"""

import numpy as np
import nibabel as nib
from scipy import ndimage

# generate random "stick" inside a volume.

xmax = 600
ymax = 300
zmax = 300

V = np.zeros((xmax, ymax, zmax))

lmin = 10
lmax = 30


n_sticks = 70

for _ in np.arange(n_sticks):
    
    x0 = int(np.random.random_sample() * (xmax-1))
    y0 = int(np.random.random_sample() * (ymax-1))
    z0 = int(np.random.random_sample() * (zmax-1))
    
    
    # x - y boundary
    boundary = [np.random.random_sample(), np.random.random_sample()]
    boundary.sort()
    print(boundary)
    V[x0, y0, z0] = 1
    
    for it in np.arange(np.random.randint(low=lmin, high=lmax+1)):
        rnd = np.random.random_sample()
        if rnd <= boundary[0]:
            x0 += 1
        elif rnd > boundary[0] and rnd <= boundary[1]:
            y0 += 1
        else:
            z0 += 1
        
        if x0 >= xmax-1 or y0 >= ymax-1 or z0 >= zmax-1:
            break
        else:
            V[x0, y0, z0] = 1
            

V = ndimage.morphology.binary_dilation(V)

file = '/home/stefan/PYTHON/HQDatasetVesselAnnot/test_noise_generation/stick_noise.nii.gz'
        
nib.save(nib.Nifti1Image(V.astype(np.uint8), np.eye(4)), file)

