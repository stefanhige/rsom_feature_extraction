#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 19:08:29 2019

@author: sgerl
"""


from pathlib import Path

import os

from classes import RSOM

from utils.get_unique_filepath import get_unique_filepath


# define folder
#origin = '/home/stefan/Documents/RSOM/Diabetes/new_data/no_surf'
origin = '/home/stefan/PYTHON/HQDatasetVesselAnnot/mat'
# origin = '/media/nas_ads_mwn/AG-Ntziachristos/RSOM_Data/RSOM_Diabetes/Stefan/allmat'
# origin = '/media/nas_ads_mwn/AG-Ntziachristos/RSOM_Data/RSOM_Diabetes/Stefan/'

# destination = '/media/nas_ads_mwn/AG-Ntziachristos/RSOM_Data/RSOM_Diabetes/Stefan/'
# destination = '/home/sgerl/Documents/PYTHON/TestDataset20190411/selection/other_preproccessing_tests/sliding_mip_6'
destination = '/home/stefan/PYTHON/HQDatasetVesselAnnot/myskin'


# mode
mode = 'list'

if mode=='dir':
    cwd = os.getcwd()
    # change directory to origin, and get a list of all files
    os.chdir(origin)
    all_files = os.listdir()
    os.chdir(cwd)
elif mode=='list':
    patterns = ['R_20190430164629']
    all_files = [os.path.basename(get_unique_filepath(origin, pat)[0]) for pat in patterns]


# extract the LF.mat files,
filenameLF_LIST = [el for el in all_files if el[-6:] == 'LF.mat']

for idx, filenameLF in enumerate(filenameLF_LIST):
    # the other ones will be automatically defined
    filenameHF = filenameLF.replace('LF.mat','HF.mat')
    #TODO: review before running
    
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
    #Obj.saveSURFACE((destination + ''), fstr = 'surf')
    
    # MIP image for quick check
    Obj.calcMIP(do_plot = False)
    Obj.saveMIP(destination, fstr = 'mip')
    
    # MIP 3D for annotation
    # Obj.calcMIP3D(do_plot = False)
    #Obj.saveMIP3D(destination, fstr = 'mip3d')
    
    # VOLUME
    Obj.normINTENSITY()
    Obj.rescaleINTENSITY(dynamic_rescale = False)
    Obj.mergeVOLUME_RGB()
    Obj.saveVOLUME(destination, fstr = 'rgb')
    
    print('Processing file', idx+1, 'of', len(filenameLF_LIST))













