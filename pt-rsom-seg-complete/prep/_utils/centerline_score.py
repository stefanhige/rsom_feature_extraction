#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  7 21:06:48 2019

@author: stefan
"""

import nibabel as nib
import numpy as np

from scipy import ndimage
from skimage import morphology
from skimage import exposure

import os

origin = '/home/stefan/PYTHON/HQDatasetVesselAnnot/vessels/R_20190605163439_HQ0003_th_corrected_rso.nii.gz'

 
# load input file
label = (nib.load(origin)).get_fdata()
label = label.astype(bool)


# fake prediction
pred = label.copy()
pred[0:200,50:100,111:300] = 1


S = morphology.skeletonize_3d(label.astype(np.uint8))
S = S.astype(bool)


# calculate centerline score
# number of pixels of sceleton inside pred / number of pixels in sceleton
cl_score = np.count_nonzero(np.logical_and(S, pred)) / np.count_nonzero(S)


# dilate label massive
# to generate hull

element = morphology.ball(5) # good value seems in between 3 and 5
element = element.astype(bool)

H = ndimage.morphology.binary_dilation(label, iterations=1, structure=element)

# 1 - number of pixels of prediction outside hull / number of pixels of prediction inside hull ? 
# or just total number of pixels of prediction
out_score = 1 - np.count_nonzero(np.logical_and(np.logical_not(H), pred)) / np.count_nonzero(pred)



img = nib.Nifti1Image(S.astype(np.uint8), np.eye(4))
nib.save(img, origin.replace('.nii.gz','_sceleton.nii.gz'))

img = nib.Nifti1Image(H.astype(np.uint8), np.eye(4))
nib.save(img, origin.replace('.nii.gz','_hull.nii.gz'))
