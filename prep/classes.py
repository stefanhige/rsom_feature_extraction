#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  6 16:44:10 2019

@author: sgerl
"""
from pathlib import Path


import scipy.io as sio
from scipy import interpolate
from scipy import ndimage
from scipy.optimize import minimize_scalar

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.cm as cm
#from mpl_toolkits.mplot3d import Axes3D
#from matplotlib.ticker import LinearLocator, FormatStrFormatter

import time


import nibabel as nib

#from skimage import data
from skimage import exposure



# CLASS FOR ONE DATASET
class RSOM():
    """
    class for preparing RSOM matlab data for layer segmentation
    """
    def __init__(self, filepathLF, filepathHF, filepathSURF='none'):
        """ create empty instance of RSOM """
        
        idx_1 = filepathLF.name.find('PAT')
        if idx_1 == -1:
            idx_1 = filepathLF.name.find('VOL')
            
        ID = filepathLF.name[idx_1:idx_1+11]
        
        self.file = self.FileStruct(filepathLF, filepathHF, filepathSURF, ID)
        
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
        # TODO: solve problem: python crashes when accessing reconParams
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
        
        
    def normINTENSITY(self, ignore_neg = True, sliding_max = False):
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
        res = minimize_scalar(self.calc_alpha, bounds=(0, 100), method='bounded')
        
        alpha = res.x
        
        P = np.dstack([self.Pl, alpha * self.Ph, np.zeros(self.Ph.shape)])
        
        # cut negative values, in order to allow rescale to uint8
        P[P < 0] = 0
        
        P = exposure.rescale_intensity(P, out_range = np.uint8)
        P = P.astype(dtype=np.uint8)
        P = exposure.rescale_intensity(P, in_range = (0.03*255, 0.3*255), out_range = np.uint8)
        
        plt.figure()
        plt.imshow(P)
        plt.title(str(self.file.ID))
        #plt.imshow(P, aspect = 1/4)
        
        plt.show()
        
        
    def calc_alpha(self, alpha):
        return np.sum(np.square(self.Pl - alpha * self.Ph))
    
    def calc_alpha_3d(self, alpha):
        return np.sum(np.square(self.Vl_split - alpha * self.Vh_split))
    
    def calcMIP_sliced(self, plot = 1):
        ''' plot maximum intensity projection along second axis '''
        
            
        # split along axis=1, 171 pixel / 9 = 19
        # get back 9 equally sized arrays
        self.Vl_split = np.split(self.Vl, 9, axis = 1)
        self.Vh_split = np.split(self.Vh, 9, axis = 1)
        
        # for every slice, perform a maximum intensity projection
        for idx in range(len(self.Vl_split)):
            self.Vl_split[idx] = np.amax(self.Vl_split[idx], axis = 1, keepdims = True)
            self.Vh_split[idx] = np.amax(self.Vh_split[idx], axis = 1, keepdims = True)
        
        
        self.Vl_split = np.concatenate(self.Vl_split, axis = 1)
        self.Vh_split = np.concatenate(self.Vh_split, axis = 1)
            
            
        # global alpha calc
        # TODO: maybe try slice-wise alpha calc?
        res = minimize_scalar(self.calc_alpha_3d, bounds=(0, 100), method='bounded')
        
        alpha = res.x
        
        # RGB stack
        self.P_sliced = np.stack([self.Vl_split, alpha * self.Vh_split, np.zeros(self.Vl_split.shape)], axis = -1)
        
        self.P_sliced[self.P_sliced < 0] = 0
        
        self.P_sliced = exposure.rescale_intensity(self.P_sliced, out_range = np.uint8)
        self.P_sliced = self.P_sliced.astype(dtype=np.uint8)
        
        # rescale intensity
        val = np.quantile(self.P_sliced,(0.8, 0.9925))
        print("Quantile", val[0], val[1], 'fixed values', round(0.03*255), 0.2*255, 'up to', 0.3*255)
        
        self.P_sliced = exposure.rescale_intensity(self.P_sliced, in_range = (val[0], val[1]), out_range = np.uint8)
    
        if plot:
            shp = self.P_sliced.shape
            P_ = self.P_sliced.copy()
            P_[:,:,-3:-1,:] = 255
            plt.figure()
            plt.imshow(P_.reshape((shp[0], shp[1]*shp[2], -1)))
            plt.show()
        
        
        # TODO: try global alpha calc, and local alpha calc and compare
        # TODO: 
        #       calculate Vl_split and Vh_split
        #       calculate alpha, either for each slice manually, or global
        #       merge to  R G B, getting 4D array (can copy this from merging function for the whole volume)
        #       save thing to nii.gz
        #       try labeling the MIP slices in ITK snap, ask Hailong how many layers we should segment
        #       
        #       Post-proc: interpolate MIP label slices back to original volume (interpolate on integer space ??)
        #
        #       this should then be the input for the 2.5D conv net, actually it's a 2d conv net
        #
        # TODO: CUT SOMEHOW TO EVEN OR UNEVEN INTEGER
        
        
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
        pl, ph = np.percentile(V, (0, 99))
        
        return exposure.rescale_intensity(V, in_range=(pl, ph))
        
        
    def cut236(self):
        ''' cut the volume to 200x300x600'''
        
        self.Vm = self.Vm[:200,:300,:600]
        
    def cut236_2channel(self):
        
        self.Vl = self.Vl[:200,:300,:600]
        self.Vh = self.Vh[:200,:300,:600]
        
        
    def saveMIP(self, fstr = ''):
        ''' save rgb maximum intensity projection volume'''
        
        # Vm is a 4-d numpy array, with the last dim holding RGB
        shape_3d = self.P_sliced.shape[0:3]
        rgb_dtype = np.dtype([('R', 'u1'), ('G', 'u1'), ('B', 'u1')])
        #self.Vm = self.P_sliced.astype('u1')
        self.P_sliced = self.P_sliced.view(rgb_dtype).reshape(shape_3d)
        img = nib.Nifti1Image(self.P_sliced, np.eye(4))
        
        mat_file = self.file.LF
        name = mat_file.name
        name = name.rstrip('LF.mat') + fstr + '.nii.gz'
        nii_file = mat_file.parents[0] / name
        
        nib.save(img, str(nii_file))
        
    def saveVOLUME(self, fstr = ''):
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
        name = name.rstrip('LF.mat') + fstr + '.nii.gz'
        nii_file = mat_file.parents[0] / name
        
        nib.save(img, str(nii_file))
        
        
    class FileStruct():
        """ helper class for data management"""
        def __init__(self, filepathLF, filepathHF, filepathSURF, ID):
            self.LF = filepathLF
            self.HF = filepathHF
            self.SURF = filepathSURF
            self.ID = ID