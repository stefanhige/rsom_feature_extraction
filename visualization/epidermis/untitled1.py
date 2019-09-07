#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  7 15:43:15 2019

@author: stefan
"""

from mayavi import mlab


import nibabel as nib
import numpy as np
from skimage import morphology, filters


filepath = '/home/stefan/PYTHON/HQDatasetVesselAnnot/input_for_layerseg/R_20190605163439_HQ0003_rgb.nii.gz'


img = nib.load(filepath)

RGB = img.get_data()
RGB = np.stack([RGB['R'], RGB['G'], RGB['B']], axis=-1)
RGB = RGB.astype(np.uint8)


red = RGB[...,0]
green = RGB[...,1]

# thresholding
red[red<170] = 0
green[green<200] = 0

red = morphology.remove_small_objects(red, min_size=100)
green = morphology.remove_small_objects(green, min_size=100)

#red = filters.gaussian(red, sigma=5, preserve_range=True)
#green = filters.gaussian(green, sigma=5, preserve_range=True)



red = red.astype(bool)
green = green.astype(bool)

# remove small objects
red = red.astype(np.uint8)
green = green.astype(np.uint8)


#dphi, dtheta = np.pi/250.0, pi/250.0
#[phi,theta] = np.mgrid[0:pi+dphi*1.5:dphi,0:2*pi+dtheta*1.5:dtheta]
#m0 = 4; m1 = 3; m2 = 2; m3 = 3; m4 = 6; m5 = 2; m6 = 6; m7 = 4;
#r = np.sin(m0*phi)**m1 + np.cos(m2*phi)**m3 + np.sin(m4*theta)**m5 + np.cos(m6*theta)**m7
#x = r*np.sin(phi)*np.cos(theta)
#y = r*np.cos(phi)
#z = r*np.sin(phi)*np.sin(theta)
#
## View it.
#s = mlab.mesh(x, y, z)
#mlab.show()


obj = mlab.contour3d(red, color=(1,0,0), opacity=0.1)

obj2 = mlab.contour3d(green, color=(0,1,0), opacity=0.1)


# load segmentation

filepath = '/home/stefan/PYTHON/HQDatasetVesselAnnot/input_for_layerseg/prediction/R_20190605163439_HQ0003_pred.nii.gz'


img = nib.load(filepath)

seg = img.get_fdata()
seg = seg.astype(np.uint8)
obj = mlab.contour3d(seg, color=(0,0,1), opacity=0.1)





#mlab.show()