#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 17:09:24 2019

@author: stefan
"""
import nibabel as nib
import numpy as np


file = '/home/stefan/PYTHON/HQDatasetVesselAnnot/vessels/R_20180117142908_VOL013_RL01_th_rso_edit.nii.gz'


file_handle = nib.load(file)
label = file_handle.get_fdata()
label = label.astype(np.uint8)


# merge label 1 and 2
print(np.amax(label))
label = label.astype(np.bool).astype(np.uint8)
print(np.amax(label))


img = nib.Nifti1Image(label, np.eye(4))
        
file = file.replace('.nii.gz','')
file = file + '_merge.nii.gz'
nib.save(img, file)



