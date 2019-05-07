#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  6 16:33:15 2019

@author: sgerl
"""

# loop over all .mat files and return histogram of z-values
# in order to determine a global cutoff for z

import os
import shutil
from pathlib import Path
import numpy as np

import matplotlib.pyplot as plt


from classes import RSOM


cwd = os.getcwd()


#origin = '/home/sgerl/Documents/171213/VOL008/alignedCorr/Recons'
origin = '/media/nas_ads_mwn/AG-Ntziachristos/RSOM_Data/RSOM_Diabetes/Stefan/allmat'
os.chdir(origin)


# list of all .mat files
all_mat = os.listdir()

os.chdir(cwd)
# extract the LF files,
all_mat_lf = [el for el in all_mat if el[-6:] == 'LF.mat']


# check if there are Surf files for all LF files
print('Missing surface files: ... ')

for curr in all_mat_lf:
    idx_1 = curr.find('_')
    idx_2 = curr.find('_', idx_1+1)
    surf = 'Surf' + curr[idx_1:idx_2+1] + '.mat'
    
    found = [el for el in all_mat if el == surf]
    
    # no surface file exists
    if not found:
        print(curr)    
    



shp_vec = np.zeros((1,3))

for idx, mat_lf in enumerate(all_mat_lf):
    
    mat_hf = mat_lf.rstrip('LF.mat') + 'HF.mat'
    idx_1 = mat_lf.find('_')
    idx_2 = mat_lf.find('_', idx_1+1)
    mat_surf = 'Surf' + mat_lf[idx_1:idx_2+1] + '.mat'
    
    # merge paths
    fullpathHF = (Path(origin) / mat_hf).resolve()
    fullpathLF = (Path(origin) / mat_lf).resolve()
    fullpathSurf = (Path(origin) / mat_surf).resolve()
    
    Obj = RSOM(fullpathLF, fullpathHF, fullpathSurf)
    
    print(idx+1, '/', len(all_mat_lf), 'trying to access:')
    print(str(fullpathLF))
    print(str(fullpathHF))
    print(str(fullpathSurf))
    
    
    Obj.readMATLAB()
    
    # accumulate dimensions
    shp_vec = np.vstack((shp_vec, np.array(Obj.Vl.shape)))


# drop first element
shp_vec = shp_vec[1:,:]


# check if all x and y are the same
if not np.any(shp_vec[:,1]-shp_vec[0,1]):
    print('all x are the same:', shp_vec[0,1])
if not np.any(shp_vec[:,2]-shp_vec[0,2]):
    print('all y are the same:', shp_vec[0,2])
    
# print histogram of z
    
zdata = shp_vec[:,0]
fig, ax = plt.subplots()
ax.hist(zdata.ravel(), bins = 100, histtype='step', color='black')
#ax.ticklabel_format(axis='y', style='scientific', scilimits=(0, 0))
#ax.set_xlabel('Pixel intensity')
#ax.set_xlim(0, 1)
#ax.set_yticks([])
    

    