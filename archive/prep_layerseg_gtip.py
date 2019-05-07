#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  1 16:37:36 2019

interpolate the ground truth labels back to the original volume size

@author: sgerl
"""
from pathlib import Path

import nibabel as nib

import numpy as np

import scipy.interpolate as interpolate

from skimage import exposure


class RSOM_MIP_IP():
    def __init__(self, filepath):
        ''' create an instance of RSOM_MIP_IP; requires path '''
        self.filepath = filepath
        
    def readNII(self):
        ''' read in the .nii.gz files specified in self.filepath'''
        
        img = nib.load(str(self.filepath))
        self.L_sliced = img.get_fdata()
        self.L_sliced = self.L_sliced.astype(np.uint8)
        
    def saveNII(self):
        ''' '''
        

        self.L = self.L.astype(np.uint8)
        img = nib.Nifti1Image(self.L, np.eye(4))
        
        file = self.filepath
        
        name = file.name
        
        name = name.rstrip('.nii.gz') + '_EXPAND' + '.nii.gz'
        nii_file = file.parents[0] / name
        
        nib.save(img, str(nii_file))
        
    def interpolate(self):
        ''' interpolate the .nii.gz label files along axis=1 '''
        
        # label volume is           depth x 9 x 333
        # need to interpolate to    depth x 171 x 333
        
        x_mip = 9
        x_rep = 171
        
        label_min = np.amin(self.L_sliced)
        label_max = np.amax(self.L_sliced)
        
        #self.L_sliced = exposure.rescale_intensity(self.L_sliced, in_range = (label_min, label_max), out_range = np.uint8)
        #self.L_sliced = self.L_sliced.astype(np.uint8)
        
        shp = self.L_sliced.shape
    
        # create grid for L_sliced
        x1 = np.arange(shp[0])
        x2 = np.linspace(np.floor(x_rep/x_mip/2), x_rep - np.ceil(x_rep/x_mip/2), num = x_mip)
        x3 = np.arange(shp[2])
        
        # dimension to be interpolated
        x2_q = np.arange(x_rep)
        
        
        #x1_q, x2_q, x3_q = (
                #x1[:, None, None], x2_q[None, :, None], x3[None, None, :])
        
        # TODO: 
        #Interpolation somehow still does not work
        # another approach, along axis=0, search for change in value,
        # and create a 2d function, based on the indices
        
        # it works, just need to stack a for loop on top, in order to 
        # process an aribitrary number of layers in z-direction
        
        # in case, label order is random, sort
        
        # determine label order of input
        
        # create synthetic extra label
        # TODO: REMOVE THIS
        #self.L_sliced[0:20,:,:] = 2
        
        L_sliced_ = self.L_sliced.copy().astype(np.int64)
        
        # cut in direction dim 0 throught middle of the volume
        L_1d = L_sliced_[:,int(shp[1]/2),int(shp[2]/2)].copy()
        
        # find the number of labels, and their indices
        # np.unique returns the labels in ascending order
        labels, idx = np.unique(L_1d, return_index = True)
        
        # we want the labels in direction dim 0, so lets sort them
        idxx = np.argsort(idx)
        labels = labels[idxx]
        
        labels = labels.astype(np.int64)
        
        n_labels = labels.size
        
        
        # check if labels are in shape: 0 1 2 3 4 already
        # TODO: check boolean expression
        if not ((labels[0] == 0) and (np.any(np.diff(labels) - 1))):
            # if not: reshape
            # add some 'large' number to the labels
            L_sliced_ += 20
            
            layer_ctr = 0
            
            for nl in np.arange(n_labels):
                L_sliced_[L_sliced_ == labels[nl] + 20] = layer_ctr
                layer_ctr += 1
                
        # INPUT: in dim 0: ascending index: label order: 0 1 2 3 4
        self.L = np.zeros((shp[0], x_rep, shp[2]))
        
        for nl in np.arange(n_labels - 1):
            surf = np.zeros((x_mip, shp[2]))
    
            for xx in np.arange(x_mip):
                for yy in np.arange(shp[2]):
                    
                    #idx = np.nonzero(np.logical_not(self.L_sliced[:, xx, yy]))
                    idx = np.nonzero(L_sliced_[:, xx, yy])
                    
                    surf[xx, yy] = idx[0][0]
                    
            fn = interpolate.interp2d(x3, x2, surf, kind = 'linear')
            surf_ip = fn(x3, x2_q)
            
            
            
            for xx in np.arange(x_rep):
                for yy in np.arange(shp[2]):
                    self.L[0:np.round(surf_ip[xx, yy]).astype(np.int), xx, yy] += 1
            
            # NEXT LABEL
            L_sliced_ -= 1
            L_sliced_[L_sliced_ < 0] = 0
            
        return self.L
    
        # RESULT: in dim 0: ascending index: label order: 4 3 2 1 0
        
        
#        self.L = interpolate.interpn(
#                (x1, x2, x3),
#                self.L_sliced,
#                (x1_q, x2_q, x3_q),
#                method='linear',
#                bounds_error = False,
#                fill_value = None)
#        
#        self.L_255 = self.L.copy()
#        
#        self.L = np.round(self.L)
#        self.L = exposure.rescale_intensity(self.L, in_range = np.uint8, out_range = (label_min, label_max))
#        
#        # TODO EROSION DILATION?
#        #scipy.ndimage.morphology.grey_closing
#        
#        print('Shape of L', self.L.shape)

    
# define filenames
filename_LIST = ['R_20171127151451_VOL002_RL01_Josefine_RSOM50_wl1_corrRGBMIP_GT.nii.gz']

#idx = 1
#filenameLF_LIST = filenameLF_LIST[idx-1:idx]
# define folder
folder = 'TestDataset20190411/selection/layerseg'


for filename in filename_LIST:
      
    
    # merge paths
    fullpath = (Path.cwd() / folder / filename).resolve()
    
    Obj = RSOM_MIP_IP(fullpath)
    
    Obj.readNII()
    L = Obj.interpolate()
    Obj.saveNII()
        
        