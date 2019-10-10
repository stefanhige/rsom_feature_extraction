#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 14 17:06:11 2019

@author: stefan
"""

import imageio
import numpy as np
import nibabel as nib
import os
import re
from skimage import filters
from skimage import transform, exposure
import matplotlib.pyplot as plt



import sys

# to make classes importable
sys.path.append('../../prep/')
from classes import RSOM

class RSOM_visualization(RSOM):
    '''
    subclass of RSOM
    for various visualization tasks
    '''

    def loadSEGMENTATION(self, filename):
        '''
        load segmentation file (can be labeled our predicted)

        '''
        self.seg = (self.loadNII(filename)).astype(np.uint8)

    def calcMIP_SEGMENTATION(self, axis=1, padding=(0, 0)):
        '''
        calculate pseudo-mip of boolean segmentation
        '''
        
        mip = np.sum(self.seg, axis=axis) >= 1

        # probably need some further adjustment on mip.shape[-1]
        print('seg mip shape:', mip.shape)
        mip = np.concatenate((np.zeros((padding[0], mip.shape[-1]), dtype=np.bool), 
                              mip.astype(np.bool),
                              np.zeros((padding[1], mip.shape[-1]), dtype=np.bool)), axis=0)
        # naming convention self.P is normal MIP 
        self.axis_P = axis
        self.P_seg = mip

    def cutMIP(self):
        '''
        cut MIP to make it fit segmentation mip, if not padding the latter
        acually need to cut volume first and then make mip?
        '''

    def mergeMIP(self, do_plot=True):
        '''
        merge MIP and MIP of segmentation with feeding into blue channel
        '''

        P_seg = self.P_seg.astype(np.float32)
        P_seg_edge = filters.sobel(P_seg)
        P_seg_edge = P_seg_edge/np.amax(P_seg_edge)

        # feed into blue channel
        # blue = 150 * P_seg + 200 * P_seg_edge
        # very light edge
        blue = 150 * P_seg + 30 * P_seg_edge
        blue[blue>255] = 255
        blue = blue.astype(np.uint8)
        self.P[:, :, 2] = blue


        # test, white background
        # P = self.P.astype(np.int64)
        # P += 100*np.ones(P.shape, dtype=np.int64)

        # print(np.amax(P
        scale = 2
        # A = np.array([[scale, 0, 0], [0, scale, 0], [0, 0, scale]])
        # self.P = ndimage.affine_transform(self.P, A)
        if scale != 1:
            
            self.P = transform.rescale(self.P, scale, order=3, multichannel=True)
            # strangely transform.rescale is not dtype consistent?
            self.P = exposure.rescale_intensity(self.P, out_range=np.uint8)
            self.P = self.P.astype(np.uint8)
        print(self.P.shape)

        # self.P[np.where((self.P == [0,0,0]).all(axis = 2))] = [255,255,255]   
        if do_plot:
            plt.figure()
            plt.imshow(self.P)
            plt.title(str(self.file.ID))
            #plt.imshow(P, aspect = 1/4)
            plt.show()
    
    def saveCombinedMIP(self, dest):
        
        no_overlay = self.P.copy()
        no_overlay[:,:,2] = 0
        
        if self.P.shape[0] > self.P.shape[1]:
            axis = 1
        else:
            axis = 0
        
        grey = 50
        
        img = np.concatenate((np.pad(no_overlay, 
                                     ((2, 2),(2, 2),(0, 0)), 
                                     mode='constant', 
                                     constant_values=grey),
                             np.pad(self.P, 
                                    ((2, 2),(2, 2),(0, 0)), 
                                    mode='constant',
                                     constant_values=grey)),
                             axis=axis)
        img = np.pad(img, 
                     ((2, 2),(2, 2),(0, 0)), 
                     mode='constant', 
                     constant_values=grey)
        
        img_file = os.path.join(dest, 'R' + 
                                   self.file.DATETIME + 
                                   self.file.ID +
                                   '_' +
                                   'combMIP_ax' +
                                   str(self.axis_P) + 
                                   '.png')
        print(str(img_file))
        
        
        imageio.imwrite(img_file, img)
        
        
 

def get_unique_filepath(path, pattern):
    
    all_files = os.listdir(path)
    # print(all_files)

    assert isinstance(pattern, str)
    occurrences = [el for el in all_files if pattern in el]
    
    if occurrences:
        # in case we are processing .mat files
        if '.mat' in occurrences[0]:
            if len(occurrences) > 2:
                raise ValueError(pattern + ' not unique!')
            else: 
                if 'LF' in occurrences[0]:
                    return os.path.join(path, occurrences[0]), os.path.join(path, occurrences[1])
                elif 'HF' in occurrences[0]:
                    return os.path.join(path, occurrences[1]), os.path.join(path, occurrences[0])
        # in case of other files (.nii.gz)
        else:
            if len(occurrences) > 1:
                raise ValueError(pattern + ' not unique!')
            else:
                return os.path.join(path, occurrences[0])
    else:
        raise Exception('No file found for \'{}\' in {}'.format(pattern, path))




file_ids = ['R_20170828154106_',
            'R_20170906132142_',
            'R_20170906141354_',
            'R_20171211150527_',
            'R_20171213135032_',
            'R_20180409164251_']  # enough string to identify an unique file

# directory with the raw matlab files
mat_dir = '/home/stefan/Documents/RSOM/Diabetes/allmat'


# directory with the segmentation files
seg_dir = '/home/stefan/fbserver_ssh/data/layerunet/for_vesnet/selection1/\
vessels/191003-00-rt_nolabel_pred/prediction/only_segmentation'

# where to put the mip
out_dir = '/home/stefan/fbserver_ssh/data/layerunet/for_vesnet/selection1/\
vessels/191003-00-rt_nolabel_pred/prediction/mip'

#file_ids = [file_ids[3]]

for file_id in file_ids:

    matLF, matHF = get_unique_filepath(mat_dir, file_id)
    print(matLF)
    
    idx_1 = matLF.find('_')
    idx_2 = matLF.find('_', idx_1+1)
    matSurf = os.path.join(mat_dir, 'Surf' + matLF[idx_1:idx_2+1] + '.mat')
    
    
    Obj = RSOM_visualization(matLF, matHF, matSurf)
    Obj.readMATLAB()
    Obj.flatSURFACE()
    
    # z=500
    Obj.cutDEPTH()
    
    seg_file = get_unique_filepath(seg_dir, file_id)
    
    z0 = int(re.search('(?<=_z)\d{1,3}(?=_)', seg_file).group())
    print('z0 = ', z0)
    
    # axis = 0
    axis = 0
    Obj.calcMIP(axis=axis, cut_z=z0)
    Obj.loadSEGMENTATION(seg_file)
    Obj.calcMIP_SEGMENTATION(axis=axis, padding=(0, 0))
    Obj.mergeMIP()
    Obj.saveCombinedMIP(out_dir)
    # axis = 1
    axis = 1
    Obj.calcMIP(axis=axis, cut_z=0)
    Obj.loadSEGMENTATION(seg_file)
    Obj.calcMIP_SEGMENTATION(axis=axis, padding=(z0, 0))
    Obj.mergeMIP()
    Obj.saveCombinedMIP(out_dir)
    
    #axis =2
    axis = 2
    Obj.calcMIP(axis=axis, cut_z=0)
    Obj.loadSEGMENTATION(seg_file)
    Obj.calcMIP_SEGMENTATION(axis=axis, padding=(z0, 0))
    Obj.mergeMIP()
    Obj.saveCombinedMIP(out_dir)



if 0:
    mip_path = '/home/stefan/PYTHON/HQDatasetVesselAnnot/out_from_prep/'
    in_filename = 'R_20190605163439_HQ0003_mip.png'

    mip_path = os.path.join(mip_path, in_filename)

    label_path = '/home/stefan/PYTHON/HQDatasetVesselAnnot/vessels/R_20190605163439_HQ0003_th_corrected_rso.nii.gz'



    # load image
    mip = imageio.imread(mip_path)


    # load label
    label = (nib.load(label_path)).get_fdata()

    z0 = 98

    mip_label = np.sum(label, axis=1) >= 1

    mip_label = np.concatenate((np.zeros((z0, mip_label.shape[-1]), dtype=np.bool), 
                                mip_label.astype(np.bool)), axis = 0)



    out_path = '/home/stefan/PYTHON/HQDatasetVesselAnnot/test_mip_overlay'
    out_filename = in_filename.replace('.png', '_ol___.png')
    out_path = os.path.join(out_path, out_filename)


    mip_label = mip_label.astype(np.float32)
    mip_label_edge = filters.sobel(mip_label)
    mip_label_edge = mip_label_edge/np.amax(mip_label_edge)

    # feed into blue channel
    blue = 150*mip_label + 200*mip_label_edge
    blue[blue>255] = 255
    blue = blue.astype(np.uint8)

    mip_overlay = mip.copy()
    mip_overlay[:, :, 2] = blue

    imageio.imwrite(out_path, mip_overlay)
