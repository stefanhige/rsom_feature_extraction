#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 19:08:29 2019

@author: sgerl
"""


from pathlib import Path


import scipy.io as sio
from scipy import interpolate
from scipy import ndimage

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import LinearLocator, FormatStrFormatter


import nibabel as nib

from skimage import data
from skimage import exposure





# NEXT STEPS
# try that quadratic intensity projection on 0...255 ?
# try the algorithm on a few more datasets, and accumulate maybe 5 .nii.gz files

# go back to CNN, and maybe can try these 5 datasets on johannes algorithm
# do the deep learning shit? what about training with synthetic datasets?


# CLASS FOR ONE DATASET
class RSOMforCNN():
    """
    class for transforming RSOM data to be suitable for CNN vessel segmentation
    """
    def __init__(self, filepathLF, filepathHF, filepathSURF='none'):
        """ create empty instance of RSOMforCNN"""
        self.file = self.FileStruct(filepathLF, filepathHF, filepathSURF)
        
    def readMATLAB(self):
        ''' read .mat files'''
        # load HF data
        self.matfileHF = sio.loadmat(self.file.HF)
        
        # extract high frequency Volume
        self.Vh = self.matfileHF['R']
        
        # load LF data
        self.matfileLF = sio.loadmat(self.file.LF)
        
        # extract low frequency Volume
        self.Vl = self.matfileLF['R']
        
        # load surface data
        self.matfileSURF = sio.loadmat(self.file.SURF)
        
    def flatSURFACE(self, override = True):
        ''' modify volumetric data in order to get a flat skin surface'''
        
        # parse surface data and dx and dy
        S = self.matfileSURF['surfSmooth']
        dx = self.matfileSURF['dx']
        dy = self.matfileSURF['dy']
        
        # create meshgrid for surface data
        xSurf = np.arange(0, np.size(S, 0)) * dx
        ySurf = np.arange(0, np.size(S, 1)) * dy
        xSurf -= np.mean(xSurf)
        ySurf -= np.mean(ySurf)
        xxSurf, yySurf = np.meshgrid(xSurf, ySurf)
    
        # create meshgrid for volume data
        # use grid step dv
        # TODO: extract from reconParams
        # TODO: solve problem: pyhton crashes when accessing reconParams
        dv = 0.012
        xVol = np.arange(0, np.size(self.Vl, 2)) * dv
        yVol = np.arange(0, np.size(self.Vl, 1)) * dv
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
        
        
        
        if not override:
            # create copy 
            self.Vl_notflat = self.Vl.copy()
            self.Vh_notflat = self.Vh.copy()
            
        # for every surface element, calculate the offset
        # and shift volume elements perpendicular to surface
        for i in np.arange(np.size(self.Vl, 1)):
            for j in np.arange(np.size(self.Vl, 2)):
                
                offs = int(-np.around(Sip[i, j]/2))
                
                #print(offs)
            
                self.Vl[:, i, j] = np.roll(self.Vl[:, i, j], offs);
                
                self.Vh[:, i, j] = np.roll(self.Vh[:, i, j], offs);
    
    
        #print(np.sum(Vl-Vlold))

        
    def plotSURFACE(self):
        ''' plot surfaceSmooth of MATLAB data'''
        
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        
        # parse surface data and dx and dy
        S = self.matfileSURF['surfSmooth']
        dx = self.matfileSURF['dx']
        dy = self.matfileSURF['dy']
        
        # create meshgrid for surface data
        xSurf = np.arange(0, np.size(S, 0)) * dx
        ySurf = np.arange(0, np.size(S, 1)) * dy
        xSurf -= np.mean(xSurf)
        ySurf -= np.mean(ySurf)
        xxSurf, yySurf = np.meshgrid(xSurf, ySurf)
        
        S = S - np.amin(S)


        # Plot the surface.
        surf = ax.plot_surface(xxSurf, yySurf, S.transpose(), cmap=cm.coolwarm,
                   linewidth=0, antialiased=False)

        # Customize the z axis.
        #ax.set_zlim(-1.01, 1.01)
        #ax.zaxis.set_major_locator(LinearLocator(10))
        #ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

        # Add a color bar which maps values to colors.
        fig.colorbar(surf, shrink=0.5, aspect=5)

        plt.show()
    
    def rescaleVOLUME(self):
        ''' rescale the volumetric data in order to get uniform grid spacing'''
        
        # ipo = 1 means linear interpolation
        ipo = 1
                
        # in reconPARAMS GRID_DZ = 3, GRID_DS = 12
        # in order to get uniform grid spacing, need to scale x and y by 4
        # or scale z, y, x by 0.5, 2, 2
        z = 0.5
        x = 2
        y = 2
    
        self.Vl = ndimage.zoom(self.Vl, (z, y, x), order = ipo)
        self.Vh = ndimage.zoom(self.Vh, (z, y, x), order = ipo)
        
        # only if the variable exists
        # adjust the size of V*_notflat
        try:
            self.Vl_notflat = ndimage.zoom(self.Vl_notflat, (z, y, x), order = ipo)
            self.Vh_notflat = ndimage.zoom(self.Vh_notflat, (z, y, x), order = ipo)
        except AttributeError:
            # donothing
            nothing = False
                
        # can't keep the old values, uses too much memory
        self.OverrideV = True
        
    def markSKIN(self):
        ''' find the approximate location of the skin surface just parallel to the image boundaries
        !! only call this method after normINTENSITY
        put a small mark at the edges of the corresponding z-layer
        '''
        
        # just use low frequency channel
        # project to 1D along z
        V1d = np.fabs(np.sum(np.sum(self.Vl, axis = 2), axis = 1))
        
        
        #plt.figure()
        #plt.plot(V1d)
       # plt.show()
        
        V1d_smooth = ndimage.filters.gaussian_filter1d(V1d, 4)
        V1d_smooth = V1d_smooth / np.amax(V1d_smooth)
        
        zSkin = np.argwhere(V1d_smooth > 0.3)[0]
    
        #cB = np.kron(np.ones((int(np.shape(self.Vl_1)[1]/2), int(np.shape(self.Vl_1)[2]/2))), np.array([[1, 0], [0, 0]]))
        
        # replace one layer with checkerboard
        try:
            self.Vm[int(zSkin),0:5,0:5] = 1
            self.Vm[int(zSkin),-6:-1,-6:-1] = 1
            self.Vm[int(zSkin),-6:-1,0:5] = 1
            self.Vm[int(zSkin),0:5,-6:-1] = 1
            
        except AttributeError:
            # Vm does not exist, mark in the individual frequency images
            self.Vl_1[int(zSkin),0:5,0:5] = 1
            self.Vl_1[int(zSkin),-6:-1,-6:-1] = 1
            self.Vl_1[int(zSkin),-6:-1,0:5] = 1
            self.Vl_1[int(zSkin),0:5,-6:-1] = 1
            
            self.Vh_1[int(zSkin),0:5,0:5] = 1
            self.Vh_1[int(zSkin),-6:-1,-6:-1] = 1
            self.Vh_1[int(zSkin),-6:-1,0:5] = 1
            self.Vh_1[int(zSkin),0:5,-6:-1] = 1
            
        
        #self.Vm[int(zSkin),:,:] = cB
        

        
    
    def normINTENSITY(self, ignore_neg = True, sliding_max = True):
        ''' normalize intensities to [0 1] '''
        
        
        
        #img = nib.Nifti1Image(self.Vl.astype(dtype = np.float32), np.eye(4))
        
        #nib.save(img,'Vol2_raw.nii')

        # use a sliding maximum filter
        if sliding_max:
            self.Vl_1 = self.labelNormalization(self.Vl)
            self.Vh_1 = self.labelNormalization(self.Vh)
        else:
            self.Vl_1 = np.true_divide(self.Vl, np.quantile(self.Vl, 1))
            self.Vh_1 = np.true_divide(self.Vh, np.quantile(self.Vh, 1))
            
            
        
        #plotSlice(self.Vh_1, 100)
        
        
        # TODO: there are still some values >1, due to boundary problems of
        # labelNormalization
        # cut above 1
        self.Vl_1[self.Vl_1 > 1] = 1
        self.Vh_1[self.Vh_1 > 1] = 1
       
        # delete negative values
        if ignore_neg:
            self.Vl_1[self.Vl_1 < 0] = 0
            self.Vh_1[self.Vh_1 < 0] = 0
        else:
            # cap below -1
            self.Vl_1[self.Vl_1 < -1] = -1
            self.Vh_1[self.Vh_1 < -1] = -1

            # move to positive values
            self.Vl_1 += 1
            self.Vh_1 += 1
    
            # scale to unity
            self.Vl_1 = self.Vl_1 / 2
            self.Vh_1 = self.Vh_1 / 2
    
         
        #plotSlice(self.Vl_1, 100)

        #plotSlice(self.Vh_1, 100)
        
    @staticmethod
    def labelNormalization(V):
        
        
        # ACTUALLY SPHERE KERNEL DOES NOT WORK
        # problem with ndimage.maximum_filter, it requires more memory
        # when used with a custom footprint
        # therefore we'll stick to a cube footprint,
        # if a sphere would be better, maybe we can define 
        # our own maximum_filter function to do the work
        
        
        # create sphere kernel
        # cube shape
        # better use uneven value
        #size = 31
        # use the middle of the image
        #center = int(size/2)
        # radius is the minimum distance to edge    
        #radius = min(center, size - center)
        
        #Y, X, Z = np.ogrid[:size, :size, :size]
        
        # create 'distance from center' cube
        #dist_from_center = np.sqrt((X - center)**2 + (Y - center)**2 + (  
        # create boolean sphere
        #kernel = dist_from_center <= radius
        
        
        #plotSlicebool(kernel, center)
        
        
        
        # kernel = np.ones((size, size, size), dtype = np.bool_)
        
        # use a maximum filter over whole volume
        # TODO: something does not work with the boundaries, ie y = 170, z = 332
        #self.Vl_maxlabel = ndimage.maximum_filter(self.Vl, footprint = kernel, mode='nearest')
        
        #self.Vl_maxlabel = ndimage.maximum_filter(self.Vl, size = 41, mode='nearest')
        
        #plotSlice(self.Vl, 100)
        #plotSlice(self.Vl_maxlabel, 100)
        
        # in areas where there is a lot of noise and almost no signal,
        # the noise gets amplified too much, therefore suggest to define a Cutoff for the maximum
        
        # calculate the maximum of the maximum label Volume
        #maxMax = np.amax(self.Vl_maxlabel)
        # and the average maximum        
        #meanMax = np.mean(self.Vl_maxlabel)
        
        # set the cutoff at ratio: 2* mean/max below mean
        #cutoff = 2 * (meanMax/maxMax) * meanMax 
        
        #self.Vl_maxlabel[self.Vl_maxlabel < cutoff] = cutoff
        
        
        #Vl_1 = np.true_divide(Vl, Vl_maxlabel)
        
        V_maxlabel = ndimage.maximum_filter(V, size = 21, mode='nearest')
        
        #plotSlice(self.Vl, 100)
        #plotSlice(self.Vl_maxlabel, 100)
        
        # in areas where there is a lot of noise and almost no signal,
        # the noise gets amplified too much, therefore suggest to define a Cutoff for the maximum
        
        # calculate the maximum of the maximum label Volume
        maxMax = np.amax(V)
        # and the average maximum        
        # meanMax = np.mean(V)
        
        # set the cutoff at ratio: 
        cutoff = 0.01 * maxMax
        
        #print(cutoff)
        
        V_maxlabel[V_maxlabel < cutoff] = cutoff
        
        V_1 = np.true_divide(V, V_maxlabel)
        
        #plotSlice(V_maxlabel, 100)
        
        return V_1

        
    def mergeVOLUME(self):
        ''' merge low frequency and high frequency data '''
        
        self.Vl_1 = self.projINTENSITY(self.Vl_1)
        self.Vh_1 = self.projINTENSITY(self.Vh_1)
        
        self.Vm = (self.Vl_1 + self.Vh_1) / 2
           
        #erg = self.eqHISTOGRAM(self.Vm)
        
        # maybe just do histogram equalization
        
        
        #self.Vm = self.nonlinearMerging(self.Vm)
        
        #plotSlice(Vm, 100)
        
    def mergeVOLUME_RGB(self):
        ''' merge low frequency and high frequency data feeding into different
        colour channels'''
        
        self.Vl_1 = self.projINTENSITY(self.Vl_1)
        self.Vh_1 = self.projINTENSITY(self.Vh_1)
        
        B = np.zeros(np.shape(self.Vl_1))
        
        
        self.Vm = np.stack([self.Vl_1, self.Vh_1, B], axis = -1)
        
        
        
    @staticmethod
    def projINTENSITY(V):
        ''' rescale intensity of input Volume'''
        
        
        #fig, ax = plt.subplots()
        
        #plotSlice(V, 100)
        
        #plot_img_and_hist(V, ax)
        
        
        # Equalization
        #Veq = exposure.equalize_hist(V)
        
        p2, p98 = np.percentile(V, (0, 99))
        #print(p2, p98)
        Vrsc = exposure.rescale_intensity(V, in_range=(p2, p98))
        
        #fig, ax = plt.subplots()
        #plot_img_and_hist(Vrsc, ax)
        #plotSlice(Vrsc, 100)
        
        #Vrsc = np.power(Vrsc, 2)

        #fig, ax = plt.subplots()
        #plot_img_and_hist(Vrsc, ax)
        #plotSlice(Vrsc, 100)
        
        #fig, ax = plt.subplots()
        #plot_img_and_hist(Veq, ax)
        #plotSlice(Veq, 100)
        
        # fig, ax = plt.subplots()
        #plot_img_and_hist(Vrsc, ax)
        #plotSlice(Vrsc, 100)
        
        return Vrsc
    
    def plotMIP(self, axis = 1):
        ''' plot maximum intensity projection along second axis '''
        
        axis = int(axis)
        if axis > 2:
            axis = 2
        if axis < 0:
            axis = 0
            
        
        # maximum intensity projection
        self.Pl = np.amax(self.Vl, axis = axis)
        self.Ph = np.amax(self.Vh, axis = axis)
        
        # calculate alpha
        alpha = np.arange(0, 100, 0.01)
        
        diff = np.zeros(alpha.size)
        
        for i in range(alpha.size):
            diff[i] = np.sum(np.square(self.Pl - alpha[i] * self.Ph))
            
        arg_min_diff = np.argmin(diff)
        
        alpha = alpha[arg_min_diff]
        
        print(alpha)
        
        P = np.dstack((self.Pl, alpha * self.Ph, np.zeros(self.Ph.shape)))
        
        # cut negative values, in order to allow rescale to uint8
        P[P < 0] = 0
        
        P = exposure.rescale_intensity(P, out_range = np.uint8)
        P = P.astype(dtype=np.uint8)
        P = exposure.rescale_intensity(P, in_range = (0.03*255, 0.3*255), out_range = np.uint8)
        
        plt.figure()
        plt.imshow(P)
        #plt.imshow(P, aspect = 1/4)
        
        plt.show()
        
        return P, alpha
    
    def plotMIP_slice(self, axis = 1):
        ''' plot maximum intensity projection along second axis '''
        
        axis = int(axis)
        if axis > 2:
            axis = 2
        if axis < 0:
            axis = 0
            
        
        # maximum intensity projection
        Vl_cut = self.Vl[:,0:20,:]
        Vh_cut = self.Vh[:,0:20,:]
        
        
        Pl = np.amax(Vl_cut, axis = axis)
        Ph = np.amax(Vh_cut, axis = axis)
        
        # calculate alpha
        alpha = np.arange(0, 100, 0.01)
        
        diff = np.zeros(alpha.size)
        
        for i in range(alpha.size):
            diff[i] = np.sum(np.square(Pl - alpha[i] * Ph))
            
        arg_min_diff = np.argmin(diff)
        
        alpha = alpha[arg_min_diff]
        
        print(alpha)
        
        P = np.dstack((Pl, alpha * Ph, np.zeros(Ph.shape)))
        
        P[P < 0] = 0
        
        P = exposure.rescale_intensity(P, out_range = np.uint8)
        P = P.astype(dtype=np.uint8)
        P = exposure.rescale_intensity(P, in_range = (0.03*255, 0.3*255), out_range = np.uint8)
        
        plt.figure()
        plt.imshow(P)
        #plt.imshow(P, aspect = 1/4)
        
        plt.show()
        
        
        
    def cut236(self):
        ''' cut the volume to 200x300x600'''
        
        self.Vm = self.Vm[:200,:300,:600]
        
        
    def cut236_2channel(self):
        
        self.Vl = self.Vl[:200,:300,:600]
        self.Vh = self.Vh[:200,:300,:600]
    
    def save(self):
        ''' save the merged Volume in one .nii file'''
        
        self.Vm = exposure.rescale_intensity(self.Vm, out_range = np.uint8)
        
        img = nib.Nifti1Image(self.Vm.astype(dtype = np.uint8), np.eye(4))
        
        mat_file = self.file.LF
        
        name = mat_file.name
        
        name = name.rstrip('LF.mat') + '.nii'
        
        
        nii_file = mat_file.parents[0] / name
        
        nib.save(img, str(nii_file))
        
        
    def save_RGB(self):
        ''' save rgb volume'''
        
        self.Vm = exposure.rescale_intensity(self.Vm, out_range = np.uint8)
        
        # Vm is a 4-d numpy array, with the last dim holding RGB
        shape_3d = self.Vm.shape[0:3]
        rgb_dtype = np.dtype([('R', 'u1'), ('G', 'u1'), ('B', 'u1')])
        self.Vm = self.Vm.astype('u1')
        self.Vm = self.Vm.view(rgb_dtype).reshape(shape_3d)
        img = nib.Nifti1Image(self.Vm, np.eye(4))
        
        mat_file = self.file.LF
        name = mat_file.name
        name = name.rstrip('LF.mat') + '.nii'
        nii_file = mat_file.parents[0] / name
        
        nib.save(img, str(nii_file))
        
        
        
    def save2(self):
        ''' save LF and HF frequency image in separate .nii files '''
        
        self.Vl_1 = exposure.rescale_intensity(self.Vl_1, out_range = np.uint8)
        self.Vh_1 = exposure.rescale_intensity(self.Vh_1, out_range = np.uint8)
        
        img = nib.Nifti1Image(self.Vl_1.astype(dtype = np.uint8), np.eye(4))
        mat_file = self.file.LF
        name = mat_file.name
        name = name.rstrip('.mat') + '.nii'
        nii_file = mat_file.parents[0] / name
        # save LF file
        nib.save(img, str(nii_file))
        
        img = nib.Nifti1Image(self.Vh_1.astype(dtype = np.uint8), np.eye(4))
        mat_file = self.file.HF
        name = mat_file.name
        name = name.rstrip('.mat') + '.nii'
        nii_file = mat_file.parents[0] / name
        # save LF file
        nib.save(img, str(nii_file))
        
    def save2_raw(self):
        ''' save LF and HF frequency image in separate .nii files '''
        
        #self.Vl_1 = exposure.rescale_intensity(self.Vl_1, out_range = np.uint8)
        #self.Vh_1 = exposure.rescale_intensity(self.Vh_1, out_range = np.uint8)
        
        img = nib.Nifti1Image(self.Vl, np.eye(4))
        mat_file = self.file.LF
        name = mat_file.name
        name = name.rstrip('.mat') + '.nii'
        nii_file = mat_file.parents[0] / name
        # save LF file
        nib.save(img, str(nii_file))
        
        img = nib.Nifti1Image(self.Vh, np.eye(4))
        mat_file = self.file.HF
        name = mat_file.name
        name = name.rstrip('.mat') + '.nii'
        nii_file = mat_file.parents[0] / name
        # save LF file
        nib.save(img, str(nii_file))
        
        
        
    class FileStruct():
        """ helper class for data management"""
        def __init__(self, filepathLF, filepathHF, filepathSURF):
            self.LF = filepathLF
            self.HF = filepathHF
            self.SURF = filepathSURF
            
            
            
            
def plotSlice(Volume, idx):
    ''' auxiliary development function '''
    plt.figure()
    
    # create a slice
    slice_ = Volume[:,idx,:];
    plt.imshow(slice_, cmap=cm.jet)
    plt.colorbar()

def plotSlicebool(Volume, idx):
    # plot the boolean data
    plt.figure()  
    # create a slice
    slice_ = Volume[:,idx,:];
    plt.imshow(slice_, cmap="Greys")
    
    
def plot_img_and_hist(image, ax, bins=256):
    """Plot an image along with its histogram and cumulative histogram.

    """
    # image = img_as_float(image)
    # ax_img, ax_hist = axes
    ax_hist = ax
    ax_cdf = ax.twinx()
    
    # Display image
    #ax.imshow(image, cmap=plt.cm.gray)
    #ax.set_axis_off()
    
    # Display histogram
    ax_hist.hist(image.ravel(), bins=bins, histtype='step', color='black')
    ax_hist.ticklabel_format(axis='y', style='scientific', scilimits=(0, 0))
    ax_hist.set_xlabel('Pixel intensity')
    ax_hist.set_xlim(0, 1)
    ax_hist.set_yticks([])
    
    # Display cumulative distribution
    img_cdf, bins = exposure.cumulative_distribution(image, bins)
    ax_cdf.plot(bins, img_cdf, 'r')
    ax_cdf.set_yticks([])


    

        


# ================ IMPORT MATLAB DATA =========================================
# `cwd`: current directory
cwd = Path.cwd()

# define filenames
filenameLF_LIST = ['R_20171127151451_VOL002_RL01_Josefine_RSOM50_wl1_corrLF.mat', #1
                   'R_20171127152019_VOL002_RL02_Josefine_RSOM50_wl1_corrLF.mat', #2
                   'R_20170726132012_PAT007_RL01_RSOM50_wl1_corrLF.mat', #3
                   'R_20170726132929_PAT007_RL02_RSOM50_wl1_corrLF.mat', #4
                   'R_20170726135613_PAT008_RL02_RSOM50_wl1_corrLF.mat', #5
                   'R_20170726140236_PAT008_RL03_RSOM50_wl1_corrLF.mat', #6
                   'R_20170726141633_PAT009_RL01_RSOM50_wl1_corrLF.mat', #7
                   'R_20170726142444_PAT009_RL02_RSOM50_wl1_corrLF.mat', #8
                   'R_20170726143750_PAT010_RL01_RSOM50_wl1_corrLF.mat', #9
                   'R_20170726144243_PAT010_RL02_RSOM50_wl1_corrLF.mat', ] #10

idx = 1
filenameLF_LIST = filenameLF_LIST[idx-1:idx]
# define folder
folder = 'TestDataset20190411/selection'


for filenameLF in filenameLF_LIST:
    # the other ones will be automatically defined
    filenameHF = filenameLF.rstrip('LF.mat') + 'HF.mat'
    # extract datetime
    idx_1 = filenameLF.find('_')
    idx_2 = filenameLF.find('_', idx_1+1)
    filenameSurf = 'Surf' + filenameLF[idx_1:idx_2+1] + '.mat'
    
    
    # merge paths
    fullpathHF = (cwd / folder / filenameHF).resolve()
    fullpathLF = (cwd / folder / filenameLF).resolve()
    fullpathSurf = (cwd / folder / filenameSurf).resolve()
    
    Vol2_01 = RSOMforCNN(fullpathLF, fullpathHF, fullpathSurf)
    
    Vol2_01.readMATLAB()
    
    # ============== SURFACE NORMALIZATION ========================================
    
    #Vol2_01.flatSURFACE()
    
    # ============ UNIFORM GRID TRANSFORMATION ====================================
    
    #Vol2_01.rescaleVOLUME()
    
    # ============= INTENSITY NORMALIZATION =======================================
    
    Vol2_01.plotMIP_slice()
    Vol2_01.plotMIP()
    #P = Vol2_01.plotMIP(axis = 1)
    #P = Vol2_01.plotMIP(axis = 2)
    
    
    
    
    #Vol2_01.plotSURFACE()
    #Vol2_01.flatSURFACE()
    #Vol2_01.rescaleVOLUME()
    #Vol2_01.plotMIP(axis = 1)
    #Vol2_01.plotMIP(axis = 2)
    #Vol2_01.plotMIP(axis = 0)
    #P = Vol2_01.plotMIP(axis = 2)
    
    
    #Vol2_01.normINTENSITY(ignore_neg = True, sliding_max = False)
    
    # TODO: try out quadratic or cubic projections f(x) = x^2 
    
    
    
    # ============= FREQUENCY CHANNEL MERGING =====================================
    
    # there is also the idea, not to merge the channels at all, but rather give it
    # as 2 channels to the CNN
    
    # but anyways, basic merging algorithm is implemented
    #RGB = Vol2_01.mergeVOLUME_RGB()
    
    #Vol2_01.markSKIN()
    
    #Vol2_01.cut236_2channel()
    #Vol2_01.save2_raw()
    

    
    #Vol2_01.save()
    #Vol2_01.save_RGB()











