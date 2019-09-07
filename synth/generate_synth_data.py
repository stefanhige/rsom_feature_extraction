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

import imageio

def noise_type_inbetween(inputVolume):
    
    inputVolume = inputVolume.astype(np.uint8)
    
    # generate filter for convolution 
    struct = morphology.ball(7)
    l = struct.shape[0]
    shell = np.zeros(struct.shape, dtype=int)
    
    for x in np.arange(l):
        for y in np.arange(l):
            nz = np.nonzero(struct[x,y,:])[0]
            if len(nz) == 1:
                shell[x,y,nz[0]] = 1
            elif len(nz) >= 2:
                shell[x,y,nz[0]] = 1
                shell[x,y,nz[-1]] = 1
                
        
    
    A = ndimage.convolve(inputVolume, shell)
    A = np.logical_and(A>=1, A<=3)

    # generate filter for convolution
    struct = ndimage.generate_binary_structure(3, 1)
    
    struct = ndimage.iterate_structure(struct, 15).astype(int)
    
    
    struct = morphology.ball(15)
    l = struct.shape[0]
    shell = np.zeros(struct.shape, dtype=int)
    
    for x in np.arange(l):
        for y in np.arange(l):
            nz = np.nonzero(struct[x,y,:])[0]
            if len(nz) == 1:
                shell[x,y,nz[0]] = 1
            elif len(nz) >= 2:
                shell[x,y,nz[0]] = 1
                shell[x,y,nz[-1]] = 1
                

    
    B = ndimage.convolve(inputVolume, shell)
    
    B = np.logical_and(B>=1, B<=3)
    
    V = np.logical_or(A, B)
    
    V = np.logical_xor(V, np.logical_and(V, inputVolume))
    
    
    # random zero out entries
    mask = np.random.random_sample(V.shape)
    
    mask = mask >= 0.9
    
    V = mask * V
    
    V_mid = ndimage.binary_dilation(V)
    V_out = ndimage.binary_dilation(V_mid)
    
    V = V.astype(np.uint8) + V_mid.astype(np.uint8) + V_out.astype(np.uint8)
    
    # add another mask?
    mask2 = np.random.random_sample(V.shape)
    
    mask2 = mask2 >= 0.5
    
    V = mask2* V
    
    
    return V

def noise_sticks(shape):
    
    xmax = shape[0]
    ymax = shape[1]
    zmax = shape[2]
    
    
    V = np.zeros((xmax, ymax, zmax))

    lmin = 10
    lmax = 40


    n_sticks = 70

    for _ in np.arange(n_sticks):
        
        x0 = int(np.random.random_sample() * (xmax-1))
        y0 = int(np.random.random_sample() * (ymax-1))
        z0 = int(np.random.random_sample() * (zmax-1))
        
        
        # x - y boundary
        boundary = [np.random.random_sample(), np.random.random_sample()]
        boundary.sort()
        #print(boundary)
        V[x0, y0, z0] = 1
        
        for it in np.arange(np.random.randint(low=lmin, high=lmax+1)):
            rnd = np.random.random_sample()
            if rnd <= boundary[0]:
                x0 += 1
            elif rnd > boundary[0] and rnd <= boundary[1]:
                y0 += 1
            else:
                z0 += 1
            
            if x0 >= xmax-1 or y0 >= ymax-1 or z0 >= zmax-1:
                break
            else:
                V[x0, y0, z0] = 1
                
    
    V = ndimage.morphology.binary_dilation(V)
    
    if np.random.random_sample() > 0.7:
            V = ndimage.morphology.binary_dilation(V)
        
    
    return V

def calc_mip(V):

    axis = 0
        
    # maximum intensity projection
    Pl = np.amax(V[...,0], axis = axis)
    Ph = np.amax(V[...,1], axis = axis)
    
    # calculate alpha
    alpha = 0.5
    
    P = np.dstack([Pl, alpha * Ph, np.zeros(Ph.shape)])
    
    # cut negative values, in order to allow rescale to uint8
    P[P < 0] = 0
    
    P = exposure.rescale_intensity(P, out_range = np.uint8)
    P = P.astype(dtype=np.uint8)
    
    # rescale intensity
    #val = np.quantile(self.P, (0.8, 0.9925))
    
    #print("Quantile", val[0], val[1], 'fixed values', round(0.03*255), 0.3*255)
    
    #self.P = exposure.rescale_intensity(self.P, in_range = (val[0], val[1]), out_range = np.uint8)
    
    return P








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
# "inbetween" noise

noise1 = noise_type_inbetween(SEG) 



# noise2 generates sticks.
noise2 = noise_sticks(A.shape)
# sticks is labeled with one
# increase intensity
noise2 = 2*noise2

# remove sticks inside vessels and a bit outside
noise2 = noise2 * np.logical_not(ndimage.binary_dilation(SEG, iterations=2))

# 
B_wnoise = B + noise1 + noise2

# gaussian background noise
gauss_background = np.random.normal(scale=0.3, size=B.shape)
gauss_background[gauss_background < 0.5] = 0
gauss_background = gauss_background * (np.logical_not(SEG)).astype(np.float)

B_wnoise += gauss_background



# foreground noise to add to red , mask by A
gauss_red = np.random.normal(scale=0.2, size=B.shape)
gauss_red = gauss_red * A.astype(np.bool).astype(np.float)

# foreground noise to add to green, mask by B
gauss_green = np.random.normal(scale=0.2, size=B.shape)
gauss_green = gauss_green * B.astype(np.bool).astype(np.float)


B_wnoise += gauss_green
# hack, usually should not be necessary
# but gauss_green makes it neccessary anyways
B_wnoise[B_wnoise>3] = 3


A += gauss_red
A[A>3] = 3


#all_noise = noise1 + noise2 + gauss_background + gauss_red + gauss_green
all_noise = noise1 + noise2 + gauss_background + 3*gauss_red + 3*gauss_green
all_noise[all_noise<0] = 0

Vm = np.stack((A, B_wnoise, np.zeros(A.shape)), axis=-1)

Vm = exposure.rescale_intensity(Vm, out_range = np.uint8)
        
shape_3d = Vm.shape[0:3]
rgb_dtype = np.dtype([('R', 'u1'), ('G', 'u1'), ('B', 'u1')])
Vm_nii = Vm.astype('u1')
Vm_nii = Vm_nii.copy().view(rgb_dtype).reshape(shape_3d)
img = nib.Nifti1Image(Vm_nii, np.eye(4))

file_ = file.replace('.nii.gz','')
file_ = file_ + '_RGB_3noise.nii.gz'
        
nib.save(img, file_)


# generate maximum intensity projection
MIP = calc_mip(Vm)

file_mip = file_.replace('.nii.gz','_mip.png')
imageio.imwrite(file_mip, MIP)



# save segmentation
file_seg = file_.replace('.nii.gz','')
file_seg = file_seg + '_l.nii.gz'
        
nib.save(nib.Nifti1Image(SEG.astype(np.uint8), np.eye(4)), file_seg)



# save noise, debug only
file_noise = file_.replace('.nii.gz','')
file_noise = file_noise + '_noise.nii.gz'
all_noise = exposure.rescale_intensity(all_noise, out_range = np.uint8)     
nib.save(nib.Nifti1Image(all_noise.astype(np.uint8), np.eye(4)), file_noise)
