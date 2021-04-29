#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 19:08:29 2019

@author: sgerl
"""


from pathlib import Path
import argparse
import os
import sys

from classes import RSOM

if __name__ == '__main__':
    sys.path.append('../')

from utils.get_unique_filepath import get_unique_filepath

def main(args):
    
    # define folder
    origin = args.mat_dir

    destination = args.output_dir
    
    if not os.path.isdir(destination):
        Exception(destination + "does not exist.")

    all_files = os.listdir(origin)

    # extract the LF.mat files,
    filenameLF_LIST = [el for el in all_files if el[-6:] == 'LF.mat']

    for idx, filenameLF in enumerate(filenameLF_LIST):
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
        
        Obj = RSOM(fullpathLF, fullpathHF, fullpathSurf)
        
        Obj.read_matlab()
        
        Obj.flat_surface()
        Obj.cut_depth()
        
        # surface for quick check
        #Obj.saveSURFACE((destination + ''), fstr = 'surf')
        
        # MIP image for quick check
        Obj.calc_mip(do_plot = False)
        Obj.save_mip(destination, fstr = 'mip')
        
        # MIP 3D for annotation
        Obj.calc_mip3d(do_plot = False)
        Obj.save_mip3d(destination, fstr = 'mip3d')
        
        # VOLUME
        Obj.norm_intensity()
        Obj.rescale_intensity(dynamic_rescale = False)
        Obj.merge_volume_rgb()
        Obj.save_volume(destination, fstr = 'rgb')
        
        print('Processing file', idx+1, 'of', len(filenameLF_LIST))

if __name__ == '__main__':
   parser = argparse.ArgumentParser(
           description="Shrink the volume for easier labeling by creating mip3d")
   parser.add_argument('--mat-dir',
           help='directory of MATLAB data',
           required=True, type=str)

   parser.add_argument('--output-dir',
           help='directory to put volumes.',
           required=True, type=str)

   main(parser.parse_args())











