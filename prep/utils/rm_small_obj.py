#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  6 17:35:03 2019

@author: stefan
"""

import nibabel as nib
import numpy as np
from skimage import morphology

# remove small objects on one dataset which I segmented before adding
# it to thresholding method

file = '/home/stefan/PYTHON/HQDatasetVesselAnnot/HQ0002_more_signal/R_20190329182252_HQ0002_th_edit.nii.gz'


file_handle = nib.load(file)
label = file_handle.get_fdata()
label = label.astype(bool)


label = morphology.remove_small_objects(label, 30)

img = nib.Nifti1Image(label.astype(np.uint8), np.eye(4))
        
file = file.replace('.nii.gz','')
file = file + '_rso.nii.gz'
nib.save(img, file)
