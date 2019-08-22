#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 15:45:39 2019

@author: stefan
"""
import numpy as np
import os
import nibabel as nib
import shutil

from dataloader import RSOMVesselDataset
from dataloader import DropBlue, ToTensor
from patch_handling import get_volume


def saveNII(V, path):
        img = nib.Nifti1Image(V, np.eye(4))
        nib.save(img, str(path))

# 1. generate random data and label files
L_dim = D_dim = (100, 100, 100)

D1 = np.random.random_sample(D_dim)
D2 = np.random.random_sample(D_dim)
D3 = np.random.random_sample(D_dim)

L1 = np.random.random_sample(L_dim)
L2 = np.random.random_sample(L_dim)
L3 = np.random.random_sample(L_dim)

D1 = D1.astype(dtype=np.float32)
D2 = D2.astype(dtype=np.float32)
D3 = D3.astype(dtype=np.float32)

L1 = L1.astype(dtype=np.float32)
L2 = L2.astype(dtype=np.float32)
L3 = L3.astype(dtype=np.float32)

# 2. generate test directory
cwd = os.getcwd()
testdir = os.path.join(cwd,'temp_test_dl')
os.mkdir(testdir)

# 3. save files to test directory
D1_name = '1_v_rgb.nii.gz'
D2_name = '2_v_rgb.nii.gz'
D3_name = '3_v_rgb.nii.gz'

L1_name = '1_v_l.nii.gz'
L2_name = '2_v_l.nii.gz'
L3_name = '3_v_l.nii.gz'

saveNII(D1, os.path.join(testdir, D1_name))
saveNII(D2, os.path.join(testdir, D2_name))
saveNII(D3, os.path.join(testdir, D3_name))

saveNII(L1, os.path.join(testdir, L1_name))
saveNII(L2, os.path.join(testdir, L2_name))
saveNII(L3, os.path.join(testdir, L3_name))

# 4. construct dataset and dataloader

divs = (2, 3, 5)
offset = (0, 0, 0)
 
# TODO transforms
set1 = RSOMVesselDataset(testdir, 
                         divs=divs, 
                         offset = offset)

# 5. draw samples and reconstruct the patches to volumes.
try:
    set1_iter = iter(set1)
    
    for file in np.arange(3):
        sample = []
        for patch in np.arange(np.prod(divs)):
            sample.append(next(set1_iter))
        
        rec_vol = get_volume(np.array(sample), divs, offset)
except:
    pass
        
        

# 6. compare to generated data.
# 7 delete test directory
        
print('delete dir')
shutil.rmtree(testdir)