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

import os

import imageio

def noise_type_inbetween(inputVolume):
    
    inputVolume = inputVolume.astype(np.uint8)
    
    # generate filter for convolution 
    struct = morphology.ball(7)
    
    # extract shell
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
                
    # convolve, and filter by result = 1..3    
    A = ndimage.convolve(inputVolume, shell)
    A = np.logical_and(A>=1, A<=3)

    # generate filter for convolution
    struct = morphology.ball(15)
    
    # extract shell
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
                
    # convolve, and filter by result = 1..3   
    B = ndimage.convolve(inputVolume, shell)
    B = np.logical_and(B>=1, B<=3)
    
    # merge together
    V = np.logical_or(A, B)
 
    # the following whole code block generates coarse-grained noise bunches

    # random zero-out entries
    mask = np.random.random_sample(V.shape)
    mask = mask >= 0.9
    V = mask * V
    
    # dilate the remaining noise
    V_mid = ndimage.binary_dilation(V)
    V_out = ndimage.binary_dilation(V_mid)
    V = V.astype(np.uint8) + V_mid.astype(np.uint8) + V_out.astype(np.uint8)
    
    # add another random mask on top
    mask2 = np.random.random_sample(V.shape)
    mask2 = mask2 >= 0.5
    
    V = mask2 * V

    return V

def noise_type_sticks(shape):
    
    xmax = shape[0]
    ymax = shape[1]
    zmax = shape[2]
    
    V = np.zeros((xmax, ymax, zmax))

    # length of the noise sticks
    lmin = 10
    lmax = 40

    # how many sticks to generate
    n_sticks = 400

    for _ in np.arange(n_sticks):
        
        # generate random starting point
        x0 = int(np.random.random_sample() * (xmax-1))
        y0 = int(np.random.random_sample() * (ymax-1))
        z0 = int(np.random.random_sample() * (zmax-1))
          
        # generate random decision boundary
        # in which direction to propagate
        boundary = [np.random.random_sample(), np.random.random_sample()]
        boundary.sort()

        V[x0, y0, z0] = 1
        
        # till random length
        for it in np.arange(np.random.randint(low=lmin, high=lmax+1)):
            
            # generate random direction sample
            # but scale with the same decision boundary for the whole stick
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
    
    # enhace stick thickness               
    V = ndimage.morphology.binary_dilation(V)
    
    # randomly enhance thickness even more
    if np.random.random_sample() > 0.7:
            V = ndimage.morphology.binary_dilation(V)
          
    return V

def calc_mip(V):

    V = V.astype(np.float)
    
    axis = 0
        
    # maximum intensity projection
    Pl = np.amax(V[...,0], axis = axis)
    Ph = np.amax(V[...,1], axis = axis)
    
#    # rescale intensites of red and green channels
#    lmin = np.amin(Pl[np.nonzero(Pl)])
#    lmax = np.amax(Pl)
#    print(lmin, lmax)
#    Pl = exposure.rescale_intensity(Pl, in_range=(200, 255), out_range=(0.3,0.6))
#    
#    hmin = np.amin(Ph[np.nonzero(Ph)])
#    hmax = np.amax(Ph)
#    print(hmin, hmax)
#    Ph = exposure.rescale_intensity(Ph, in_range=(200, 255), out_range=(0.3,0.6))
#    
#    Pl = np.sqrt(Pl)
#    Ph = np.sqrt(Ph)
#    # calculate alpha
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



def rsom_style(label):
    """
    load synthetic vessel data, boolean image. apply several transformation
    to make it look like rsom
    """
    
    # Red channel
    # erode 2 times, and remove separated vessels
    A = ndimage.binary_erosion(label, iterations=2)
    A = morphology.remove_small_objects(A, min_size=100)
    
    # dilate remaining vessels again,
    # copies can be stacked together for different intensities
    A_inner = ndimage.binary_dilation(A, iterations=1)
    A_mid = ndimage.binary_dilation(A_inner, iterations=1)
    A = ndimage.binary_dilation(A_mid, iterations=1)
    
    # Green channel   
    # erode 2 times, remove sepearated vessels
    B = ndimage.binary_erosion(label, iterations=2)
    
    B = morphology.skeletonize_3d(B)
    # dilate 2 times, but remove small objects inbetween,
    # so we get slightly more vessels than Red channel
    B = ndimage.binary_dilation(B, iterations=1)
    B = morphology.remove_small_objects(B, min_size=200)   
    B_inner = ndimage.binary_erosion(B, iterations=1)
    
    # additionally, increase size of "Green inside Red" channel
    B_in_A = np.logical_and(A, B) 
    B_in_A = ndimage.binary_dilation(B_in_A, iterations=1)
    
    # merge back to B
    B = np.logical_or(B_in_A, B)
 
    # extract Green not in Red
    B_not_in_A = np.logical_and(np.logical_not(A), B)
    
    # save segmentation
    #file_seg = file.replace('.nii.gz', '_debug2.nii.gz')      
    #nib.save(nib.Nifti1Image(B_not_in_A_2.astype(np.uint8), np.eye(4)), file_seg)
    #return 0
    
    # stack together for different intensities
    # sum is 3
    A = 0.75 * A_inner.astype(np.float) + 1.5 * A_mid.astype(np.float) + 0.75*A.astype(np.float)
    
    # intensity of "only" green generally lower, subtract B_not_in_A
    B = 0.5 * B_inner.astype(np.float) + 2.5 * B.astype(np.float) - B_not_in_A
    
    
    # this looks like a good stage to generate the segmentation here!
    SEG = np.logical_or(A, B)
    
    # add some noise 
    # "inbetween" noise
    noise1 = noise_type_inbetween(SEG) 
    
    # noise2 generates sticks.
    noise2 = noise_type_sticks(A.shape)
    # sticks are scaled to 1
    # increase intensity
    noise2 = 2 * noise2
    
    # remove sticks inside dilated hull of the vessels
    noise2 = noise2 * np.logical_not(ndimage.binary_dilation(SEG, iterations=2))
    
    # add the noise to Green channel
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
    # edit: but now gauss_green makes it neccessary anyways
    B_wnoise[B_wnoise>3] = 3
    
    A += gauss_red
    A[A>3] = 3
    
    # stack all noise together for debug output
    # all_noise = noise1 + noise2 + gauss_background + 3*gauss_red + 3*gauss_green
    # all_noise[all_noise<0] = 0
    
    # just in case, remove negative values, which could come from the noise
    A[A<0] = 0
    B_wnoise[B_wnoise<0] = 0
    
    
    # generate RGB volume
    Vm = np.stack((A, B_wnoise, np.zeros(A.shape)), axis=-1)

    Vm = exposure.rescale_intensity(Vm, out_range = np.uint8)
    
    return Vm, SEG 



root_dir = '/home/stefan/PYTHON/synthDataset/seg'
dest_dir = '/home/stefan/PYTHON/synthDataset/rsom_style'
# change directory to origin, and get a list of all files
all_files = os.listdir(root_dir)
all_files.sort()

# extract the n.nii.gz files
filenames = [el for el in all_files if el[-7:] == '.nii.gz' and el[:-7].isdigit()]

for filename in filenames: 
    
    origin = os.path.join(root_dir, filename)
    dest = os.path.join(dest_dir, filename)
    print('Processing ', filename)
    
    # load input file
    label = (nib.load(origin)).get_fdata()
    label = label.astype(bool)

    Vm, SEG = rsom_style(label)

    # generate maximum intensity projection
    # MIP = calc_mip(Vm)
    # file_mip = file.replace('.nii.gz','_mip.png')
    # imageio.imwrite(file_mip, MIP)

    # save rgb 
    shape_3d = Vm.shape[0:3]
    rgb_dtype = np.dtype([('R', 'u1'), ('G', 'u1'), ('B', 'u1')])
    Vm_nii = Vm.astype('u1')
    Vm_nii = Vm_nii.copy().view(rgb_dtype).reshape(shape_3d)
    img = nib.Nifti1Image(Vm_nii, np.eye(4))
    nib.save(img, dest.replace('.nii.gz', '_v_rgb.nii.gz'))

    # save segmentation
    seg = nib.Nifti1Image(SEG.astype(np.uint8), np.eye(4))
    nib.save(seg, dest.replace('.nii.gz','_v_l.nii.gz'))
    
    
    
    
    
    
    
    
