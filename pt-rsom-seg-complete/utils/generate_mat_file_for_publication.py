#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  4 14:15:53 2020

@author: stefan
"""

import scipy.io as sio

# VOL018_RL02

fileHF = '/home/stefan/PYTHON/HQDatasetVesselAnnot/mat/R_20190430164629_Stefan_LeftVolarArterialCuff_off_1_RSOM50_wl1_corrHF.mat'
fileLF = '/home/stefan/PYTHON/HQDatasetVesselAnnot/mat/R_20190430164629_Stefan_LeftVolarArterialCuff_off_1_RSOM50_wl1_corrLF.mat'
fileSURF = '/home/stefan/PYTHON/HQDatasetVesselAnnot/mat/Surf_20190430164629_.mat'

# load HF data
matfileHF = sio.loadmat(fileHF)
        
# extract high frequency Volume
Vh = matfileHF['R']

outHF = {'R': Vh}

sio.savemat('/home/stefan/example_mat/R_20200101000000_EX0001_HF.mat', outHF, appendmat=False, format='5',
            do_compression=True)
        
# load LF data
matfileLF = sio.loadmat(fileLF)
        
# extract low frequency Volume
Vl = matfileLF['R']

outLF = {'R': Vl}
sio.savemat('/home/stefan/example_mat/R_20200101000000_EX0001_LF.mat', outLF, appendmat=False, format='5',
            do_compression=True)

matfileSURF = sio.loadmat(fileSURF)

# parse surface data and dx and dy
S = matfileSURF['surfSmooth']
dx = matfileSURF['dx']
dy = matfileSURF['dy']

outSURF = {'surfSmooth': S,
           'dx':dx,
           'dy':dy}
sio.savemat('/home/stefan/example_mat/Surf_20200101000000_.mat', outSURF, appendmat=False, format='5',
            do_compression=True)