#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 19:08:29 2019

@author: sgerl
"""


from pathlib import Path

import os

from classes import RSOM


#import scipy.io as sio
#from scipy import interpolate
#from scipy import ndimage
#from scipy.optimize import minimize_scalar

#import numpy as np

#import matplotlib.pyplot as plt
#import matplotlib.cm as cm
#from mpl_toolkits.mplot3d import Axes3D
#from matplotlib.ticker import LinearLocator, FormatStrFormatter

# ================ IMPORT MATLAB DATA =========================================
# `cwd`: current directory
#cwd = Path.cwd()

# define filenames
#filenameLF_LIST = ['R_20171127151451_VOL002_RL01_Josefine_RSOM50_wl1_corrLF.mat', #1
#                   'R_20171127152019_VOL002_RL02_Josefine_RSOM50_wl1_corrLF.mat', #2
#                   'R_20170726132012_PAT007_RL01_RSOM50_wl1_corrLF.mat', #3
#                   'R_20170726132929_PAT007_RL02_RSOM50_wl1_corrLF.mat', #4
#                   'R_20170726135613_PAT008_RL02_RSOM50_wl1_corrLF.mat', #5
#                   'R_20170726140236_PAT008_RL03_RSOM50_wl1_corrLF.mat', #6
#                   'R_20170726141633_PAT009_RL01_RSOM50_wl1_corrLF.mat', #7
#                   'R_20170726142444_PAT009_RL02_RSOM50_wl1_corrLF.mat', #8
#                   'R_20170726143750_PAT010_RL01_RSOM50_wl1_corrLF.mat', #9
#                   'R_20170726144243_PAT010_RL02_RSOM50_wl1_corrLF.mat', ] #10

#idx = 1
#filenameLF_LIST = filenameLF_LIST[idx-1:idx]

# define folder


#origin = '/home/sgerl/Documents/PYTHON/TestDataset20190411/selection'

origin = '/media/nas_ads_mwn/AG-Ntziachristos/RSOM_Data/RSOM_Diabetes/Stefan/allmat'

destination = '/home/sgerl/Documents/RSOM/Diabetes/fullDataset/layer_seg'


cwd = os.getcwd()

# change directory to origin, and get a list of all files
os.chdir(origin)
all_files = os.listdir()
os.chdir(cwd)


# extract the LF.mat files,
filenameLF_LIST = [el for el in all_files if el[-6:] == 'LF.mat']


for idx, filenameLF in enumerate(filenameLF_LIST):
    # the other ones will be automatically defined
    filenameHF = filenameLF.rstrip('LF.mat') + 'HF.mat'
    # extract datetime
    idx_1 = filenameLF.find('_')
    idx_2 = filenameLF.find('_', idx_1+1)
    filenameSurf = 'Surf' + filenameLF[idx_1:idx_2+1] + '.mat'
    
    
    # merge paths
    fullpathHF = (Path(origin) / filenameHF).resolve()
    fullpathLF = (Path(origin) / filenameLF).resolve()
    fullpathSurf = (Path(origin) / filenameSurf).resolve()
    
    Obj = RSOM(fullpathLF, fullpathHF, fullpathSurf)
    
    Obj.readMATLAB()
    
    Obj.flatSURFACE()
    Obj.cutDEPTH()
    
    # surface for quick check
    Obj.saveSURFACE((destination + '/surf'), fstr = 'surf')
    
    # MIP image for quick check
    Obj.calcMIP(do_plot = False)
    Obj.saveMIP(destination, fstr = 'mip')
    
    # MIP 3D for annotation
    Obj.calcMIP3D(do_plot = False)
    Obj.saveMIP3D(destination, fstr = 'mip3d')
    
    # VOLUME
    Obj.normINTENSITY()
    Obj.rescaleINTENSITY(dynamic_rescale = False)
    Obj.mergeVOLUME_RGB()
    Obj.saveVOLUME(destination, fstr = 'rgb')
    
    print('Processing file', idx+1, 'of', len(filenameLF_LIST))
    
    












