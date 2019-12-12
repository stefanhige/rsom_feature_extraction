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

from scipy.ndimage import morphology

import sys

# to make classes importable
if __name__ == '__main__':
    sys.path.append('../../')
from prep.classes import RSOM

def _overlay(data, seg, alpha=0.5, colour=[255,255,255]):
    seg_mask = seg.copy()
    seg_mask = np.dstack((seg_mask, seg_mask, seg_mask)).astype(np.bool)
    seg_rgb = np.dstack((colour[0]*seg, colour[1]*seg, colour[2]*seg))
   
    ol = (alpha * seg_mask.astype(np.uint8) * data) + \
              (np.logical_not(seg_mask).astype(np.uint8) * data) + \
              (1-alpha) * seg_rgb
          
    ol[ol>255] = 255
    ol[ol<0] = 0
    ol = ol.astype(np.uint8)
    return ol

class RsomVisualization(RSOM):
    '''
    subclass of RSOM
    for various visualization tasks
    '''

    def load_seg(self, filename):
        '''
        load segmentation file (can be labeled our predicted)

        '''
        return (self.loadNII(filename)).astype(np.uint8)

    def calc_mip_ves_seg(self, seg, axis=1, padding=(0, 0)):
        '''
        calculate pseudo-mip of boolean segmentation
        '''
        seg = self.load_seg(seg)

        mip = np.sum(seg, axis=axis) >= 1

        # probably need some further adjustment on mip.shape[-1]
        print('seg mip shape:', mip.shape)
        mip = np.concatenate((np.zeros((padding[0], mip.shape[-1]), dtype=np.bool), 
                              mip.astype(np.bool),
                              np.zeros((padding[1], mip.shape[-1]), dtype=np.bool)), axis=0)
        # naming convention self.P is normal MIP 
        self.axis_P = axis
        self.P_seg = mip

    def calc_mip_lay_seg(self, seg, axis=1, padding=(0, 0)):

        seg = self.load_seg(seg)

        mip = np.sum(seg, axis=axis) >= 1

        print('seg mip shape:', mip.shape)
        mip = np.concatenate((np.zeros((padding[0], mip.shape[-1]), dtype=np.bool), 
                              mip.astype(np.bool),
                              np.zeros((padding[1], mip.shape[-1]), dtype=np.bool)), axis=0)
        # naming convention self.P is normal MIP 
        self.axis_P = axis
        self.P_seg = mip

    def merge_mip_ves(self, do_plot=True):
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
        
        self.P_overlay = self.P.copy().astype(np.float32)
        self.P_overlay[:, :, 2] += blue
        self.P_overlay[self.P>255] = 255

        # test, white background
        # P = self.P.astype(np.int64)
        # P += 100*np.ones(P.shape, dtype=np.int64)

        # print(np.amax(P
        # A = np.array([[scale, 0, 0], [0, scale, 0], [0, 0, scale]])
        # self.P = ndimage.affine_transform(self.P, A)

        # self.P[np.where((self.P == [0,0,0]).all(axis = 2))] = [255,255,255]   
        if do_plot:
            plt.figure()
            plt.imshow(self.P)
            plt.title(str(self.file.ID))
            #plt.imshow(P, aspect = 1/4)
            plt.show()
    
    def merge_mip_lay(self, do_plot=True):
        '''
        merge MIP and MIP of segmentation with feeding into blue channel
        '''

        P_seg = self.P_seg.astype(np.float32)
        P_seg_edge = filters.sobel(P_seg)
        P_seg_edge = P_seg_edge/np.amax(P_seg_edge)
        P_seg_edge = morphology.binary_dilation(P_seg_edge)
           

        # feed into blue channel
        # blue = 150 * P_seg + 200 * P_seg_edge
        # very light edge
        blue = 150 * P_seg + 30 * P_seg_edge
        blue[blue>255] = 255
       
        # P_seg_edge_ma = np.ma.masked_array(P_seg_edge,mask=P_seg_edge==0)
                
        self.P_overlay[:, :, 2] += 255*P_seg_edge
        self.P_overlay[:, :, 0] += 100*P_seg_edge
        self.P_overlay[:, :, 1] += 100*P_seg_edge
        self.P_overlay[self.P>255] = 255


        # test, white background
        # P = self.P.astype(np.int64)
        # P += 100*np.ones(P.shape, dtype=np.int64)

        # print(np.amax(P
        # A = np.array([[scale, 0, 0], [0, scale, 0], [0, 0, scale]])
        # self.P = ndimage.affine_transform(self.P, A)

        # self.P[np.where((self.P == [0,0,0]).all(axis = 2))] = [255,255,255]   
        if do_plot:
            plt.figure()
            plt.imshow(self.P)
            plt.title(str(self.file.ID))
            #plt.imshow(P, aspect = 1/4)
            plt.show()

    def _rescale_mip(self, scale):

        self.P_overlay = self.P_overlay.astype(np.uint8)
        
        if scale != 1:
            
            self.P = transform.rescale(self.P, scale, order=3, multichannel=True)
            # strangely transform.rescale is not dtype consistent?
            self.P = exposure.rescale_intensity(self.P, out_range=np.uint8)
            self.P = self.P.astype(np.uint8)
            
            self.P_overlay = transform.rescale(self.P_overlay, scale, order=3, multichannel=True)
            # strangely transform.rescale is not dtype consistent?
            self.P_overlay = exposure.rescale_intensity(self.P_overlay, out_range=np.uint8)
            self.P_overlay = self.P_overlay.astype(np.uint8)
   
    def return_mip(self, scale=2):

        self._rescale_mip(scale)

        return self.P, self.P_overlay

    def save_comb_mip(self, dest, scale=2):
        
        self._rescale_mip(scale)
        
        if self.P.shape[0] > self.P.shape[1]:
            axis = 1
        else:
            axis = 0
        
        grey = 50
        
        img = np.concatenate((np.pad(self.P, 
                                     ((2, 2),(2, 2),(0, 0)), 
                                     mode='constant', 
                                     constant_values=grey),
                             np.pad(self.P_overlay, 
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
        
        
class RsomVisualization_white(RsomVisualization):
    
    #def merge_mip_ves(self, do_plot=True):
    #    '''
    #    merge MIP and MIP of segmentation with feeding into blue channel
    #    '''
        
    #    self.P_overlay = _overlay(self.P, self.P_seg.astype(np.float32),
    #                              alpha=0.6)
        
    #    self.P_overlay[self.P>255] = 255
        
    #    if do_plot:
    #        plt.figure()
    #        plt.imshow(self.P)
    #        plt.title(str(self.file.ID))
    #        #plt.imshow(P, aspect = 1/4)
    #        plt.show()
    
    def merge_mip_lay(self, do_plot=True):
        '''
        merge MIP and MIP of segmentation with feeding into blue channel
        '''
        self.P_overlay = _overlay(self.P_overlay, self.P_seg.astype(np.float32),
                                  alpha=0.5)
        
        self.P_overlay[self.P>255] = 255
        
        if do_plot:
            plt.figure()
            plt.imshow(self.P)
            plt.title(str(self.file.ID))
            #plt.imshow(P, aspect = 1/4)
            plt.show()
    




def mip_label_overlay(file_ids, dirs, plot_epidermis=False):
    """ docstring
    """

    mat_dir = dirs['in']
    seg_dir_lay = dirs['layer']
    seg_dir_ves = dirs['vessel']
    out_dir = dirs['out']

    if isinstance(file_ids, str):
        
        file_ids = [file_ids]

    elif file_ids is None:
        
        # get file_ids from vesselseg directory
        file_ids = os.listdir(seg_dir_ves)
        file_ids = [id_[:id_.find('_', 2)] for id_ in file_ids]

    for file_id in file_ids:
        mip_label_overlay1(file_id, 
                           dirs, 
                           plot_epidermis=plot_epidermis, 
                           axis=-1, 
                           return_img=False)

def mip_label_overlay1(file_id, dirs, plot_epidermis=False, axis=-1, return_img=False):
    """ axis=-1 means all axes
    """

    mat_dir = dirs['in']
    seg_dir_lay = dirs['layer']
    seg_dir_ves = dirs['vessel']
    out_dir = dirs['out']

    matLF, matHF = get_unique_filepath(mat_dir, file_id)
    
    _, matLF_ = os.path.split(matLF)
    idx_1 = matLF_.find('_')
    idx_2 = matLF_.find('_', idx_1+1)
    matSurf = os.path.join(mat_dir, 'Surf' + matLF_[idx_1:idx_2+1] + '.mat')
    print('SURF file:', matSurf) 
    
    Obj = RsomVisualization_white(matLF, matHF, matSurf)
    Obj.readMATLAB()
    Obj.flatSURFACE()
    
    # z=500
    Obj.cutDEPTH()
    
    seg_file_ves = get_unique_filepath(seg_dir_ves, file_id)
    seg_file_lay = get_unique_filepath(seg_dir_lay, file_id)
    z0 = int(re.search('(?<=_z)\d{1,3}(?=_)', seg_file_ves).group())
    print('z0 = ', z0)
    
    if axis == -1 or axis == 0:
        # axis = 0
        # this is the top view
        axis_ = 0
        Obj.calcMIP(axis=axis_, do_plot=False, cut_z=z0)
        Obj.calc_mip_ves_seg(seg=seg_file_ves, axis=axis_, padding=(0, 0))
        Obj.merge_mip_ves(do_plot=False)
        if return_img:
            return Obj.return_mip()
        else:
            Obj.save_comb_mip(out_dir)
    if axis == -1 or axis == 1:
        # axis = 1
        axis_ = 1
        Obj.calcMIP(axis=axis_, do_plot=False, cut_z=0)
        Obj.calc_mip_ves_seg(seg=seg_file_ves, axis=axis_, padding=(z0, 0))
        Obj.merge_mip_ves(do_plot=False)
        if plot_epidermis:
            Obj.calc_mip_lay_seg(seg=seg_file_lay, axis=axis_, padding=(0, 0))
            Obj.merge_mip_lay(do_plot=False)
        if return_img:
            return Obj.return_mip()
        else:
            Obj.save_comb_mip(out_dir)
    
    if axis == -1 or axis == 2:
        axis_ = 2
        Obj.calcMIP(axis=axis_, do_plot=False, cut_z=0)
        Obj.calc_mip_ves_seg(seg=seg_file_ves, axis=axis_, padding=(z0, 0))
        Obj.merge_mip_ves(do_plot=False)
        if plot_epidermis:
            Obj.calc_mip_lay_seg(seg=seg_file_lay, axis=axis_, padding=(0, 0))
            Obj.merge_mip_lay(do_plot=False)
        if return_img:
            return Obj.return_mip()
        else:
            Obj.save_comb_mip(out_dir)

    
if __name__ == '__main__':

    file_ids = ['R_20180409171854_VOL018_RL02',
                'R_20170830135455_PAT032_RL01',
                'R_20180409165847_VOL016_RL02',
                'R_20180409172754_VOL019_RL02',
                'R_20171127155723_VOL004_RL01',
                'R_20170911152919_PAT045_RL01',
                'R_20170911151516_PAT044_RL01',
                'R_20170809135901_PAT020_RL01',
                'R_20171211153606_PAT059_RL01',
                'R_20180409170556_VOL017_RL01',
                'R_20170906132603_PAT040_RL02',
                'R_20170809131529_PAT018_RL01',
                'R_20171113152204_PAT055_RL01',
                'R_20170906141825_PAT042_RL02',
                'R_20170809133530_PAT019_RL01',
                'R_20180409171554_VOL018_RL01',
                'R_20171211152025_PAT058_RL01',
                'R_20171211152425_PAT058_RL02',
                'R_20170807153617_PAT014_RL01',
                'R_20170807160848_PAT016_RL01']  # enough string to identify an unique file

    # directory with the raw matlab files
    mat_dir = '/home/stefan/Documents/RSOM/Diabetes/processableDataset/mat'


    # directory with the segmentation files
    seg_dir_ves = '/home/stefan/Documents/RSOM/Diabetes/processableDataset/selection/threshold_segmentation'

    seg_dir_lay = '/home/stefan/fbserver_ssh/data/pipeline/processableDataset/results/tmp/layerseg_out'
    # where to put the mip
    out_dir = '/home/stefan/Documents/RSOM/Diabetes/processableDataset/selection/mip_threshold'

    dirs = {'in': mat_dir,
            'layer': seg_dir_lay,
            'vessel': seg_dir_ves,
            'out': out_dir }

    mip_label_overlay(file_ids, dirs, plot_epidermis=True)

