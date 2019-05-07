#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  3 15:38:48 2019

@author: sgerl
"""

# copy the data from nas ads mwn 

import os
import shutil


cwd = os.getcwd()

ALL_FILES = []


#origin = '/home/sgerl/Documents'
origin = '/media/nas_ads_mwn/AG-Ntziachristos/RSOM_Data/RSOM_Diabetes/!!_Diabetes'
os.chdir(origin)


folders_date = os.listdir()

# remove all folders that do not represent a date
# in origin, there are folders named YYMMDD, eg 190420
# only consider these folders


folders_date = [el for el in folders_date if el.isnumeric() and len(el) == 6]

# for all the folders
for folders_date_it in folders_date:
    os.chdir(folders_date_it)
    print(os.getcwd())
    
    # get all the PATxxx or VOLxxx folders
    folders_pat = os.listdir()
    
    # remove the others
    folders_pat = [el for el in folders_pat if (el[0:3] == 'VOL' or el[0:3] == 'PAT') and len(el) == 6]
    
    path_pat = os.getcwd() 
    # for all PATxxx and VOLxxx folders
    for folders_pat_it in folders_pat:
        recons = 'alignedCorr/Recons'
        path_mat = os.path.join(origin, folders_date_it, folders_pat_it, recons) 
        
        try:
            os.chdir(path_mat)
        except:
            print('no alignedCorr/recons folder')
        else:
            print(os.getcwd())
            
            # NOW IN THE .MAT LEVEL
            files = os.listdir()
            files = [el for el in files if el[-10:] == 'corrLF.mat' or el[-10:] =='corrHF.mat' or el[0:4] == 'Surf']

            for file_it in files:
                ALL_FILES.append(os.path.join(path_mat, file_it))
            
            os.chdir(path_pat)
        
    os.chdir(origin)
        
os.chdir(cwd)

print(os.getcwd())

# DO YOU WANT TO COPY ALL THE FILES?
destination = '/media/nas_ads_mwn/AG-Ntziachristos/RSOM_Data/RSOM_Diabetes/Stefan/allmat'

DO_COPY = 1
if DO_COPY:
    for idx, cf in enumerate(ALL_FILES):
        shutil.copy(cf, destination)
        print('copying file:', cf)
        
        if not((idx+1)%10):
            print('copied', idx+1, 'of', len(ALL_FILES), 'files')
        




