#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 10 09:54:28 2019

@author: sgerl
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  1 16:37:36 2019

interpolate the ground truth labels back to the original volume size

@author: sgerl
"""
from pathlib import Path

import nibabel as nib

import os

import numpy as np

import scipy.interpolate as interpolate

from skimage import exposure

from classes import RSOM_mip_interp
 
# define filenames
filename_LIST = ['R_20170724150057_PAT001_RL01_mip3d_l.nii.gz']


origin = '/home/sgerl/Documents/RSOM/Diabetes/fullDataset/layer_seg'
#origin = '/media/nas_ads_mwn/AG-Ntziachristos/RSOM_Data/RSOM_Diabetes/Stefan/allmat'
destination = '/home/sgerl/Documents/RSOM/Diabetes/fullDataset/layer_seg'

cwd = os.getcwd()

# change directory to origin, and get a list of all files
os.chdir(origin)
all_files = os.listdir()
os.chdir(cwd)

# extract the _mip_3d_l files
rstr = 'mip3d_l.nii.gz'

filename_LIST = [el for el in all_files if el[-len(rstr):] == rstr]


for filename in filename_LIST:
      
    
    # merge paths
    fullpath = (Path(origin) / filename).resolve()
    
    Obj = RSOM_mip_interp(fullpath)
    
    Obj.readNII()
    L = Obj.interpolate()
    Obj.saveNII(destination)