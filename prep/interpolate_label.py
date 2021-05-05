#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  1 16:37:36 2019

interpolate the ground truth labels back to the original volume size

@author: sgerl
"""
from pathlib import Path
import os
import argparse
from classes import RSOM_mip_interp
 

def main(args):
    origin = args.label_dir
    destination = args.output_dir
    
    # extract the _mip_3d_l files
    rstr = args.annotated_endswith
    all_files = os.listdir(origin)
    
    filename_list = [el for el in all_files if el[-len(rstr):] == rstr]
    
    if not filename_list:
        print("Didn't find any files in ", origin)
    for idx, filename in enumerate(filename_list):
        
        print('Processing file', idx+1, 'of', len(filename_list))
          
        # merge paths
        fullpath = os.path.join(origin, filename)
        
        obj = RSOM_mip_interp(Path(fullpath))
        
        obj.readNII()
        obj.interpolate()
        obj.saveNII(destination)
        
if __name__ == '__main__':
    
   # directory of annotated data
   label_dir = "/some/path"
   
   # directory to put interpolated labels
   output_dir = "/some/other/path"
   
   parser = argparse.ArgumentParser(
           description="interpolate the ground truth labels back to the original volume size")
   parser.add_argument('--label-dir',
           help='directory of annotated data',
           required=False, type=str)
   parser.add_argument('--output-dir',
           help='directory to put interpolated labels',
           required=False, type=str)
   parser.add_argument('--annotated-endswith',
           help='string pattern to identify annotated data',
          required=False, type=str, default='mip3d_l.nii.gz')
   args = parser.parse_args()

   # command line options overwrite dirs
   if not args.label_dir:
       args.label_dir = label_dir
   if not args.output_dir:
       args.output_dir = output_dir

   main(args)



