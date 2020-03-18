#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 25 17:21:21 2020

@author: stefan
"""
import nibabel as nib
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch
import os
# from mpl_toolkits.mplot3d import Axes3D

import matplotlib.cm as cm
from scipy.ndimage.filters import gaussian_filter1d 

from scipy.ndimage.filters import uniform_filter1d


def load_seg(path):
    img = nib.load(path)
    return img.get_data()
    
def load_rgb(path):
    img = nib.load(path)
    data = img.get_data()
    data = np.stack([data['R'], data['G'], data['B']], axis=-1)
    return data

def save_nii(V, path, fstr=None):
    
    if fstr is not None:
        if not '.nii.gz' in fstr:
            fstr += '.nii.gz'
        path = os.path.join(path, fstr)
    
    #V = V.astype(np.uint8)
    img = nib.Nifti1Image(V, np.eye(4))
    nib.save(img, path)
    
def extract_surf(P):    
    Pproj = np.sum(P, axis=2)
    
    Pproj_ = gaussian_filter1d(Pproj, sigma=7, axis=1)
    
    Maxima = np.argmax(Pproj_, axis=0)
    surf1 = np.zeros((P.shape[1], P.shape[2]))
    surf2 = np.zeros_like(surf1)
    
    zerorough=0
    for y in np.arange(P.shape[1]):
        for x in np.arange(P.shape[2]):
            nz = np.nonzero(P[:, y, x])
            nz = nz[0]
            #print(nz)
            # only valid if we captured the whole thing
            if len(nz) == 0:    
                surf1[y,x] = Maxima[y] #+ int(np.random.normal(scale=3))
                surf2[y,x] = Maxima[y] #+ int(np.random.normal(scale=3))
                zerorough += 1
                # surf1[y,x] = np.nan
                # surf2[y,x] = np.nan
                # print('no epidermis, value at', x, y, 'set to', Maxima[y])
            else:
                if nz[0] == 0:
                    # print('inside the layer!',x,y)
                    surf1[y,x] = Maxima[y] #+ int(np.random.normal(scale=3))
                    surf2[y,x] = Maxima[y] #+ int(np.random.normal(scale=3))
                    pass
                else:
                    # TODO check if holes
                    cutoff = 15
                    if np.any(np.diff(nz)>cutoff):
                        nzidx = np.nonzero((np.diff(nz)>cutoff -1).astype(np.bool))
                        nzidx = nzidx[0]
                        if len(nzidx) > 1:
                            #TODO these cases are ignored 
                            start = nz[0]
                            end = nz[-1]
                            # print(nz)
                            # print(nzidx)
                        elif nzidx < 0.2*len(nz):
                            # print('take last part')
                            # print(nz)
                            # print(nzidx)
                            start = nz[nzidx+1]
                            end = nz[-1]
                            # print(start, end)
                        elif nzidx > 0.8*len(nz):
                            # print('take first part')
                            # print(nz)
                            # print(nzidx)
                            start = nz[0]
                            end = nz[nzidx]
                            # print(start, end)
                        else:
                            start = nz[0]
                            end = nz[-1]
                    else:
                        start = nz[0]
                        end = nz[-1]
                    surf1[y,x] = start
                    surf2[y,x] = end
                    # surf1[y,x] = nz[0]
                    # surf2[y,x] = nz[-1]
    return surf1, surf2, zerorough                 
    
def rms_roughness(surf, window=5):
    mv_avg_surf = uniform_filter1d(surf, window, axis=1)
    
    # res_surf = np.sqrt(np.mean(np.power(surf - mv_avg_surf, 2),axis=1))
    res_surf = np.mean(np.abs(surf-mv_avg_surf),axis=1)
    # print(res_surf.shape)
    return res_surf
 
def plot_surface(S, dest, title=''):
    '''
    plot the surfaceData used for the normalization
    mainly for debugging purposes
    '''
    
    plt.ioff()
    plt.figure()
    
    Surf = S-np.amin(S)
    
    plt.imshow(Surf, cmap=cm.jet)
    plt.colorbar()
    plt.title(title) 
    plt.savefig(dest)
        
    plt.close()

window=5
print('window',window)


#dirs = ['/home/stefan/data/layerunet/miccai/fcn/7layer/200209-00-FCN_BCE',
#        '/home/stefan/data/layerunet/miccai/fcn/7layer/200210-00-FCN_BCE_S1',
#        '/home/stefan/data/layerunet/miccai/fcn/7layer/200210-01-FCN_BCE_S10',
#        '/home/stefan/data/layerunet/miccai/fcn/7layer/200210-02-FCN_BCE_S100']

dirs = ['/home/stefan/data/layerunet/miccai/200201-04-BCE',
         '/home/stefan/data/layerunet/miccai/200202-02-BCE_S_1',
         '/home/stefan/data/layerunet/miccai/200202-03-BCE_S_10',
         '/home/stefan/data/layerunet/miccai/200202-04-BCE_S_100',
         '/home/stefan/data/layerunet/miccai/200202-05-BCE_S_1000',
         '/home/stefan/data/layerunet/miccai/200203-02-BCE_S_2000',
         '/home/stefan/data/layerunet/miccai/200204-00-BCE_S_2500',
         '/home/stefan/data/layerunet/miccai/200204-01-BCE_S_2800']

# dirs = ['/home/gerlstefan/data/layerunet/miccai/200201-04-BCE',
#         '/home/gerlstefan/data/layerunet/miccai/200201-05-BCE_W',
#         '/home/gerlstefan/data/layerunet/miccai/200201-06-BCE_W_S_1',
#         '/home/gerlstefan/data/layerunet/miccai/200201-07-BCE_W_S_10',
#         '/home/gerlstefan/data/layerunet/miccai/200202-00-BCE_W_S_100',
#         '/home/gerlstefan/data/layerunet/miccai/200202-01-BCE_W_S_1000'
#         ]
for loc in dirs:
    files = os.listdir(loc)
    files = [el for el in files if "_pred.nii.gz" in el]
    P = []
    if 1:
        for f in files:
            P.append(load_seg(os.path.join(loc,f)))
        P = np.concatenate(P, axis=1)
        surf1, surf2, n_holes = extract_surf(P)
        # save surfaces as images
        plot_surface(surf1, os.path.join(loc,'surf1.png')) 
        plot_surface(surf2, os.path.join(loc,'surf2.png')) 
        
        surf1_rms = rms_roughness(surf1,window=window)
        surf2_rms = rms_roughness(surf2,window=window)
        rms = np.concatenate((surf1_rms, surf2_rms), axis=0)
        print(os.path.basename(loc),'holes', n_holes, 'roughness', np.mean(rms))
    if 0:    
        for f in files:
            P = load_seg(os.path.join(loc, f))
            surf1, surf2, n_holes = extract_surf(P)
        
            surf1_rms = rms_roughness(surf1,window=window)
            surf2_rms = rms_roughness(surf2,window=window)
            rms = np.concatenate((surf1_rms, surf2_rms), axis=0)
            print(os.path.basename(loc), f, 'roughness', np.mean(rms))

