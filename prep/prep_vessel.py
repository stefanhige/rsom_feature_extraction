#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 13:58:44 2019

@author: sgerl
"""

from pathlib import Path

import os

from classes import RSOM_vessel


# define folder


origin = '/home/sgerl/Documents/PYTHON/TestDatasetVessel/mat'
origin_layer = '/home/sgerl/Documents/PYTHON/TestDatasetVessel/layer_pred'

# origin = '/media/nas_ads_mwn/AG-Ntziachristos/RSOM_Data/RSOM_Diabetes/Stefan/allmat'
# origin = '/media/nas_ads_mwn/AG-Ntziachristos/RSOM_Data/RSOM_Diabetes/Stefan/'


# destination = '/media/nas_ads_mwn/AG-Ntziachristos/RSOM_Data/RSOM_Diabetes/Stefan/'
# destination = '/home/sgerl/Documents/PYTHON/TestDataset20190411/selection/other_preproccessing_tests/sliding_mip_6'
destination = '/home/sgerl/Documents/PYTHON/TestDatasetVessel/output'


cwd = os.getcwd()

# change directory to origin, and get a list of all files
os.chdir(origin)
all_files = os.listdir()
os.chdir(cwd)


# extract the LF.mat files,
filenameLF_LIST = [el for el in all_files if el[-6:] == 'LF.mat']


for idx, filenameLF in enumerate(filenameLF_LIST):
    
    #if idx >= 1:
    #    break
    # the other ones will be automatically defined
    filenameHF = filenameLF.replace('LF.mat','HF.mat')
    
    # extract datetime
    idx_1 = filenameLF.find('_')
    idx_2 = filenameLF.find('_', idx_1+1)
    filenameSurf = 'Surf' + filenameLF[idx_1:idx_2+1] + '.mat'
    
    
    # merge paths
    fullpathHF = (Path(origin) / filenameHF).resolve()
    fullpathLF = (Path(origin) / filenameLF).resolve()
    fullpathSurf = (Path(origin) / filenameSurf).resolve()
    
    Obj = RSOM_vessel(fullpathLF, fullpathHF, fullpathSurf)
    
    Obj.readMATLAB()
    
    Obj.flatSURFACE()
    Obj.cutDEPTH()
    
    # surface for quick check
    # Obj.saveSURFACE((destination + ''), fstr = 'surf')
    
    # MIP image for quick check
    # Obj.calcMIP(do_plot = False)
    # Obj.saveMIP(destination, fstr = 'mip')
    
    # MIP 3D for annotation
    # Obj.calcMIP3D(do_plot = False)
    #Obj.saveMIP3D(destination, fstr = 'mip3d')
    
    # cut epidermis
    # Obj.cutLAYER(origin_layer, fstr='layer_pred.nii.gz')
    
    # VOLUME
    Obj.normINTENSITY()
    Obj.rescaleINTENSITY(dynamic_rescale = False)
    
    Obj.mergeVOLUME_RGB()
    Obj.saveVOLUME(destination, fstr = 'v_TEST_rgb')
    
    print('Processing file', idx+1, 'of', len(filenameLF_LIST))