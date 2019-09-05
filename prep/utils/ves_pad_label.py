#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 17:18:08 2019

@author: stefan
"""
import nibabel as nib
import numpy as np

# this was only needed for one file, 
# R_20190605163439_HQ0003_th_edit_merge.nii.gz
# because i did annotation on bad threshold image (border problem)
# now is fixed, so don't need it anymore


file = '/home/stefan/PYTHON/HQDatasetVesselAnnot/vessels/R_20190605163439_HQ0003_th_edit_merge.nii.gz'


file_handle = nib.load(file)
label = file_handle.get_fdata()
label = label.astype(np.uint8)


label = np.pad(label[1:-1,1:-1,1:-1], 1, mode='edge')


img = nib.Nifti1Image(label, np.eye(4))
        
file = file.replace('.nii.gz','')
file = file + '_pad.nii.gz'
nib.save(img, file)