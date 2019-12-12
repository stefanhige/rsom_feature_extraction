#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 14:56:59 2019

@author: stefan
"""
import sys
import os
import shutil
# to make classes importable
if __name__ == '__main__':
    sys.path.append('../')
    
from utils.get_unique_filepath import get_unique_filepath

origin = '/home/stefan/Documents/RSOM/Diabetes/processableDataset/mip'

cwd = os.getcwd()
# change directory to origin, and get a list of all files
os.chdir(origin)
all_files = os.listdir()
os.chdir(cwd)


# all_files = [all_files[0]]

origin_mat = '/home/stefan/Documents/RSOM/Diabetes/allmat'
destination = '/home/stefan/Documents/RSOM/Diabetes/processableDataset/mat'

for file in all_files:
    
    fileLF, fileHF = get_unique_filepath(origin_mat, file[:22])
    
    idx_1 = file.find('_')
    idx_2 = file.find('_', idx_1+1)
    fileSurf = os.path.join(origin_mat, 'Surf' + file[idx_1:idx_2+1] + '.mat')
    
    for f in [fileLF, fileHF, fileSurf]:
        shutil.copy(f,destination)
    
    