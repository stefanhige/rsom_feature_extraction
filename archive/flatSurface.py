#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 14:04:18 2019

@author: sgerl
"""

from pathlib import Path
import scipy.io as sio

from scipy import interpolate
from scipy import ndimage
import numpy as np

import matplotlib.pyplot as plt

# colormap
import matplotlib.cm as cm
#import nibabel as nib

def flatSurffn(Vl, Vh, matfileSurf):
    
    # parse surface data and dx and dy
    S = matfileSurf['surfSmooth']
    dx = matfileSurf['dx']
    dy = matfileSurf['dy']
    
    # create meshgrid for surface data
    xSurf = np.arange(0, np.size(S, 0)) * dx
    ySurf = np.arange(0, np.size(S, 1)) * dy
    xSurf -= np.mean(xSurf)
    ySurf -= np.mean(ySurf)
    xxSurf, yySurf = np.meshgrid(xSurf, ySurf)

    # create meshgrid for volume data
    # use grid step dv
    # TODO: extract from reconParams
    dv = 0.012
    xVol = np.arange(0, np.size(Vl, 2)) * dv
    yVol = np.arange(0, np.size(Vl, 1)) * dv
    xVol -= np.mean(xVol)
    yVol -= np.mean(yVol)
    xxVol, yyVol = np.meshgrid(xVol, yVol)
    
    # create interpolation function
    #fn = interpolate.interp2d(ySurf, xSurf, S, kind = 'cubic')
    #Sip = fn(yVol, xVol)

    # try another interpolation function, supposed to be faster
    fn = interpolate.RectBivariateSpline(xSurf, ySurf, S)
    Sip = fn(xVol, yVol)
    
    # subtract mean
    Sip -= np.mean(Sip)
    
    # flip
    Sip = Sip.transpose()
    
    Vlold = Vl;
    
    Vl = Vl.copy()
    Vh = Vh.copy()
    
    # for every surface element, calculate the offset
    # and shift volume elements perpendicular to the surface
    for i in np.arange(np.size(Vl, 1)):
        for j in np.arange(np.size(Vl, 2)):
            
            offs = int(-np.around(Sip[i, j]/2))
            
            #print(offs)
        
            Vl[:, i, j] = np.roll(Vl[:, i, j], offs);
            
            Vh[:, i, j] = np.roll(Vh[:, i, j], offs);


    #print(np.sum(Vl-Vlold))
    return Vl, Vh
    
    
    
    
    

    


# `cwd`: current directory
cwd = Path.cwd()

filenameHF = 'R_20171127151451_VOL002_RL01_Josefine_RSOM50_wl1_corrHF.mat'
filenameLF = 'R_20171127151451_VOL002_RL01_Josefine_RSOM50_wl1_corrLF.mat'
filenameSurf = 'Surf_20171127151451_.mat'

folder = 'TestDataset20190411/Vol2'

fullpathHF = cwd / folder / filenameHF
fullpathLF = cwd / folder / filenameLF
fullpathSurf = cwd / folder / filenameSurf

# load hf data
matfileHF = sio.loadmat(fullpathHF.resolve())

# extract high frequency Volume
Vh = matfileHF['R']

# load lf data
matfileLF = sio.loadmat(fullpathLF.resolve())

# extract low frequency Volume
Vl = matfileLF['R']

# load surface data
matfileSurf = sio.loadmat(fullpathSurf.resolve())

# call function to calculate new Volumes with a flat surface
Vl_flat, Vh_flat = flatSurffn(Vl, Vh, matfileSurf)

# check if it worked the same as in the MATLAB script

filenameLF_Flat = 'R_20171127151451_VOL002_RL01_Josefine_RSOM50_wl1_corrLF_Flat.mat'
# load hf data
matfileLF_Flat = sio.loadmat((cwd / folder / filenameLF_Flat).resolve())

Vl_flat_ref = matfileLF_Flat['R']

diff = Vl_flat - Vl_flat_ref

#plt.figure()
#plt.imshow(diff[100,:,:])
#plt.colorbar()

#print(np.amax(diff))


# calculate maxI projection

Vl_proj = np.amax(Vl, axis=1)

Vl_flat_proj = np.amax(Vl_flat, axis=1)

plt.figure()
plt.imshow(Vl_proj)

plt.figure()
plt.imshow(Vl_flat_proj)
           
# TODO: choose some interpolation parameter for equal grid step in every direction
# OR leave data as it is, but probably need to retrain CNN on different shaped data
# data aspect ratio is 4 : 1
# either scale up dimension 1 and 2 by 4, OR reduce dimension 0 by 4, OR compromise

Vl_flat_proj_toscale = ndimage.zoom(Vl_flat_proj,(1, 4))

plt.figure()
plt.imshow(Vl_flat_proj_toscale)

# linear interpolation
Vl_flat_toscale = ndimage.zoom(Vl_flat,(0.5, 2, 2), order = 3)



