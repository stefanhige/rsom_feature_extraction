#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  6 19:06:10 2019

@author: stefan
"""

# thinning of vessels up to a certain cutoff
import nibabel as nib
import numpy as np

from scipy import ndimage
from skimage import morphology
from skimage import exposure


file = '/home/stefan/PYTHON/HQDatasetVesselAnnot/test_noise_generation/1.nii.gz'


file_handle = nib.load(file)
label = file_handle.get_fdata()
label = label.astype(bool)

# R channel

A = ndimage.binary_erosion(label.copy(), iterations=2)
A = morphology.remove_small_objects(A, min_size=100)

# at each iteration, decrease intensity
A_inner = ndimage.binary_dilation(A, iterations=1)
A_mid = ndimage.binary_dilation(A_inner.copy(), iterations=1)
A = ndimage.binary_dilation(A_mid.copy(), iterations=1)

# G channel

B = ndimage.binary_erosion(label.copy(), iterations=2)
B = morphology.skeletonize_3d(B)
B = ndimage.binary_dilation(B, iterations=1)
B = morphology.remove_small_objects(B, min_size=200)

B_inner = ndimage.binary_erosion(B.copy(), iterations=1)

# additionally, increase size of G channel by one inside R channel
B_in_A = np.logical_and(A.copy(), B.copy())

B_in_A = ndimage.binary_dilation(B_in_A, iterations=1)

# merge back to B

B = np.logical_or(B_in_A, B)


# extract Green not in Red

B_not_in_A = np.logical_xor(B, B_in_A)




# stack together for different intensities
# sum is 3
A = 0.75 * A_inner.astype(np.float) + 1.5 * A_mid.astype(np.float) + 0.75*A.astype(np.float)

# intensity of only green generally lower
B = 0.5 * B_inner.astype(np.float) + 2.5 * B.astype(np.float) - B_not_in_A



# this looks like a good stage to generate the segmentation here!

SEG = np.logical_or(A, B)


# add noise 




Vm = np.stack((A, B, np.zeros(A.shape)), axis=-1)

Vm = exposure.rescale_intensity(Vm, out_range = np.uint8)
        
shape_3d = Vm.shape[0:3]
rgb_dtype = np.dtype([('R', 'u1'), ('G', 'u1'), ('B', 'u1')])
Vm = Vm.astype('u1')
Vm = Vm.copy().view(rgb_dtype).reshape(shape_3d)
img = nib.Nifti1Image(Vm, np.eye(4))

file_ = file.replace('.nii.gz','')
file_ = file_ + '_RGB3_inside_larger_intensity_mod.nii.gz'
        
nib.save(img, file_)


# save segmentation
file_seg = file_.replace('.nii.gz','')
file_seg = file_seg + '_l.nii.gz'
        
nib.save(nib.Nifti1Image(SEG.astype(np.uint8), np.eye(4)), file_seg)
