#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 10 17:08:14 2019

@author: sgerl
"""

from classes import RSOM

from pathlib import Path

import os

import nibabel as nib

import numpy as np

from skimage import transform

#origin = '/home/sgerl/Documents/PYTHON/TestDataset20190411/selection'

#origin = '/media/nas_ads_mwn/AG-Ntziachristos/RSOM_Data/RSOM_Diabetes/Stefan/allmat'
#origin = '/media/nas_ads_mwn/AG-Ntziachristos/RSOM_Data/RSOM_Diabetes/Stefan/'


#destination = '/media/nas_ads_mwn/AG-Ntziachristos/RSOM_Data/RSOM_Diabetes/Stefan/'

#destination = '/home/sgerl/Documents/RSOM/Diabetes/fullDataset/layer_seg'


origin = '/home/sgerl/Documents/RSOM/myskin'

destination = origin

# extract the LF.mat files,
filenameLF = 'R_20190430164629_Stefan_LeftVolarArterialCuff_off_1_RSOM50_wl1_corrLF.mat'



# the other ones will be automatically defined
filenameHF = filenameLF.replace('LF.mat','HF.mat')
#TODO: review before running

# extract datetime
idx_1 = filenameLF.find('_')
idx_2 = filenameLF.find('_', idx_1+1)
filenameSurf = 'Surf' + filenameLF[idx_1:idx_2+1] + '.mat'


# merge paths
fullpathHF = (Path(origin) / filenameHF).resolve()
fullpathLF = (Path(origin) / filenameLF).resolve()
fullpathSurf = (Path(origin) / filenameSurf).resolve()

Obj = RSOM(fullpathLF, fullpathHF, fullpathSurf)

Obj.readMATLAB()

Obj.flatSURFACE()
Obj.cutDEPTH()

# MIP image for quick check
Obj.calcMIP(do_plot = True)
Obj.calcMIP3D()

# sliding mip
smip = Obj.slidingMIP(axis = 0)

smip_fine = np.zeros((0, 2*smip.shape[1], 2*smip.shape[2], 3))

for i in range(smip.shape[0]):
    sl = np.expand_dims(transform.resize(smip[i,:,:,:], (2*smip.shape[1],2*smip.shape[2]), order = 1), 0)

    smip_fine = np.concatenate((smip_fine, sl), axis=0)


#save rgb maximum intensity projection volume
# Vm is a 4-d numpy array, with the last dim holding RGB
shape_3d = smip.shape[0:3]
rgb_dtype = np.dtype([('R', 'u1'), ('G', 'u1'), ('B', 'u1')])
#self.Vm = self.P_sliced.astype('u1')
smip_ = smip.view(rgb_dtype).reshape(shape_3d)
img = nib.Nifti1Image(smip_, np.eye(4))

# generate Path object
destination = Path(destination)

# generate filename
nii_file = (destination / 'sliding_mip.nii_0.nii.gz').resolve()
print(str(nii_file))
nib.save(img, str(nii_file))






#import imageio
#
#reader = imageio.get_reader('imageio:cockatoo.mp4')
#fps = reader.get_meta_data()['fps']
#
#writer = imageio.get_writer('~/cockatoo_gray.mp4', fps=fps)
#
#for im in reader:
#    writer.append_data(im[:, :, 1])
#writer.close()