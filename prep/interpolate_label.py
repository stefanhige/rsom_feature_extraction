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
import argparse
from classes import RSOM_mip_interp
 

def main(args)
    origin = args['label-dir']
    destination = args['output-dir']
    
    # extract the _mip_3d_l files
    rstr = args['annotated-endswith']
    all_files = os.listdir(origin)
    
    filename_list = [el for el in all_files if el[-len(rstr):] == rstr]
    
    
    for filename in filename_list:
          
        # merge paths
        fullpath = os.path.join(origin, filename)
        
        obj = RSOM_mip_interp(fullpath)
        
        obj.readNII()
        obj.interpolate()
        obj.saveNII(destination)
        
if __name__ == '__main__':
   parser = argparse.ArgumentParser()
   parser.add_argument('--label-dir',
           help='directory of annotated data',
           required=True, type=str)

   parser.add_argument('--output-dir',
           help='directory to put interpolated labels',
           required=True, type=str)

   parser.add_argument('--annotated-endswith',
           help='string pattern to identify annotated data',
           required=False, type=str, default='mip3d_l.nii.gz')



