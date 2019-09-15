#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  6 16:44:10 2019

@author: sgerl
"""
from pathlib import Path

import os


import scipy.io as sio
from scipy import interpolate
from scipy import ndimage
from scipy.optimize import minimize_scalar

import imageio

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.cm as cm
#from mpl_toolkits.mplot3d import Axes3D
#from matplotlib.ticker import LinearLocator, FormatStrFormatter

import time


import nibabel as nib

#from skimage import data
from skimage import exposure
from skimage import morphology



# CLASS FOR ONE DATASET
class RSOM():
    """
    class for preparing RSOM matlab data for layer and vessel segmentation
    """
    def __init__(self, filepathLF, filepathHF, filepathSURF='none'):
        """
        create empty instance of RSOM
        """
        # if the filepaths are strings, generate PosixPath objects
        filepathLF = Path(filepathLF)
        filepathHF = Path(filepathHF)
        filepathSURF = Path(filepathSURF)

        # extact datetime number
        print(filepathLF.name)
        idx_1 = filepathLF.name.find('_')
        idx_2 = filepathLF.name.find('_', idx_1+1)
        DATETIME = filepathLF.name[idx_1:idx_2+1]
        
        # extract the 3 digit id + measurement string eg PAT001_RL01
        idxID = filepathLF.name.find('PAT')

        if idxID == -1:
            idxID = filepathLF.name.find('VOL')

            
        if idxID is not -1:
            ID = filepathLF.name[idxID:idxID+11]
        else:
            # ID is different, extract string between Second "_" and third "_"
            # usually its 6 characters long
            idx_3 = filepathLF.name.find('_', idx_2+1)
            ID = filepathLF.name[idx_2+1:idx_3]


        
        self.layer_end = None
        
        
        self.file = self.FileStruct(filepathLF, filepathHF, filepathSURF, ID, DATETIME)
        
    def readMATLAB(self):
        '''
        read .mat files
        '''
        # load HF data
        self.matfileHF = sio.loadmat(self.file.HF)
        
        # extract high frequency Volume
        self.Vh = self.matfileHF['R']
        
        # load LF data
        self.matfileLF = sio.loadmat(self.file.LF)
        
        # extract low frequency Volume
        self.Vl = self.matfileLF['R']
        
        # load surface data
        try:
            self.matfileSURF = sio.loadmat(self.file.SURF)
        except:
            print(('WARNING: Could not load surface data, placing None type in'
                   'surface file. Method flatSURFACE is not going to be applied!!'))
            self.matfileSURF = None
        
    def flatSURFACE(self, override = True):
        '''
        modify volumetric data in order to get a flat skin surface
        options:
            override = True. If False, Volumetric data of the unflattened
            Skin will be saved.
        '''
        if self.matfileSURF is not None:
            
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
        
            # generate interpolation function
            fn = interpolate.RectBivariateSpline(xSurf, ySurf, S)
            Sip = fn(xVol, yVol)
            
            # subtract mean
            Sip -= np.mean(Sip)
            
            # flip, to fit the grid
            Sip = Sip.transpose()
            
            # save to Obj
            self.Sip = Sip
            
            if not override:
                # create copy 
                self.Vl_notflat = self.Vl.copy()
                self.Vh_notflat = self.Vh.copy()
        
            # for every surface element, calculate the offset
            # and shift volume elements perpendicular to surface
            for i in np.arange(np.size(self.Vl, 1)):
                for j in np.arange(np.size(self.Vl, 2)):
                    
                    offs = int(-np.around(Sip[i, j]/2))
                    
                    # TODO: why not replace with zeros?
                    self.Vl[:, i, j] = np.roll(self.Vl[:, i, j], offs);
                    self.Vh[:, i, j] = np.roll(self.Vh[:, i, j], offs);
                    
                    # replace values rolled inside epidermis with zero
                    if offs < 0:
                        self.Vl[offs:, i, j] = 0
                        self.Vh[offs:, i, j] = 0
        
    def plotSURFACE(self):
        '''
        plot the surfaceData used for the normalization
        It's called "surfSmooth and extracted from the MATLAB data
        '''
        
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
        '''
        rescale the volumetric data Vl, Vh in order to get 
        uniform grid spacing
        '''
        # interpolationOrder
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
        
    def normINTENSITY(self, ignore_neg = True, sliding_max = False):
        '''
        normalize intensities to [0 1]
        options:
            ignore_neg = True. Cut negative intensities.
            sliding_max = False. Use adaptive maximum filter.
        '''
        
        # use a sliding maximum filter
        if sliding_max:
            self.Vl_1 = self.labelNormalization(self.Vl)
            self.Vh_1 = self.labelNormalization(self.Vh)
        else:
            self.Vl_1 = np.true_divide(self.Vl, np.amax(self.Vl))
            self.Vh_1 = np.true_divide(self.Vh, np.amax(self.Vh))
            
            
        # there might be still some values >1, due to boundary problems of
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
        
    def rescaleINTENSITY(self, dynamic_rescale = False):
        '''
        rescale intensities, quadratic, crop
        '''
        
        
        if dynamic_rescale:
            #rescale intensity
            val = np.quantile(self.Vl_1, (0, 0.99))
            print('quantile Vl', val)
            self.Vl_1 = exposure.rescale_intensity(self.Vl_1, in_range = (val[0], val[1]))
            
            
            val = np.quantile(self.Vh_1, (0, 0.99))
            print('quantile Vh', val)
            self.Vh_1 = exposure.rescale_intensity(self.Vh_1, in_range = (val[0], val[1]))
            
            self.Vl_1 = self.Vl_1**2
            self.Vh_1 = self.Vh_1**2
            
            
            val = np.quantile(self.Vl_1, (0.8, 1))
            print('quantile Vl', val)
            self.Vl_1 = exposure.rescale_intensity(self.Vl_1, in_range = (val[0], val[1]))
            
            
            val = np.quantile(self.Vh_1, (0.8, 1))
            print('quantile Vh', val)
            self.Vh_1 = exposure.rescale_intensity(self.Vh_1, in_range = (val[0], val[1]))
            
        else:
            #static
            
            # first dataset has these settings:
            
#            self.Vl_1 = exposure.rescale_intensity(self.Vl_1, in_range = (0, 0.2))
#            self.Vh_1 = exposure.rescale_intensity(self.Vh_1, in_range = (0, 0.1))
#            
#            self.Vl_1 = self.Vl_1**2
#            self.Vh_1 = self.Vh_1**2
#            
#            self.Vl_1 = exposure.rescale_intensity(self.Vl_1, in_range = (0.05, 1))
#            self.Vh_1 = exposure.rescale_intensity(self.Vh_1, in_range = (0.02, 1))
            
            self.Vl_1 = exposure.rescale_intensity(self.Vl_1, in_range = (0, 0.2))
            self.Vh_1 = exposure.rescale_intensity(self.Vh_1, in_range = (0, 0.1))
            
            self.Vl_1 = self.Vl_1**2
            self.Vh_1 = self.Vh_1**2
            
            self.Vl_1 = exposure.rescale_intensity(self.Vl_1, in_range = (0.05, 1))
            self.Vh_1 = exposure.rescale_intensity(self.Vh_1, in_range = (0.02, 1))
            

        #print('min:', np.amin(self.Vl_1))
        #print('max:', np.amax(self.Vl_1))
        
    def calcMIP(self, axis = 1, do_plot = True, cut_z=0):
        '''
        plot maximum intensity projection along specified axis
        options:
            axis = 0,1,2. Axis along which to project
            do_plot = True. Plot into figure
            cut     = cut from volume in z direction
                      needed for on top view without epidermis
            
        '''
        
        axis = int(axis)
        if axis > 2:
            axis = 2
        if axis < 0:
            axis = 0
            
        # maximum intensity projection
        self.Pl = np.amax(self.Vl[cut_z:,...], axis = axis)
        self.Ph = np.amax(self.Vh[cut_z:,...], axis = axis)
        
        # calculate alpha
        res = minimize_scalar(self.calc_alpha, bounds=(0, 100), method='bounded')
        alpha = res.x
        
        self.P = np.dstack([self.Pl, alpha * self.Ph, np.zeros(self.Ph.shape)])
        
        # cut negative values, in order to allow rescale to uint8
        self.P[self.P < 0] = 0
        
        self.P = exposure.rescale_intensity(self.P, out_range = np.uint8)
        self.P = self.P.astype(dtype=np.uint8)
        
        # rescale intensity
        val = np.quantile(self.P, (0.8, 0.9925))
        
        #print("Quantile", val[0], val[1], 'fixed values', round(0.03*255), 0.3*255)
        
        self.P = exposure.rescale_intensity(self.P, in_range = (val[0], val[1]), out_range = np.uint8)
        
        if do_plot:
            plt.figure()
            plt.imshow(self.P)
            plt.title(str(self.file.ID))
            #plt.imshow(P, aspect = 1/4)
            plt.show()
    
    def slidingMIP(self, axis=1):
        '''
        generate volume of mip's in a sliding fashion
        on the basis of Vl_1 and Vh_1
        '''
        axis = int(axis)
        if axis > 2:
            axis = 2
        if axis < 0:
            axis = 0
        
        shp = self.Vl_1.shape
        # print('shape before', self.Vl_1.shape, self.Vh_1.shape)
        
        window = 6
        
        assert not window % 2, 'window should be even, otherwise strange things may happen'
        
        if axis == 1:
            Pl = np.zeros((shp[0], 0, shp[2]))
            Ph = np.zeros((shp[0], 0, shp[2]))
        elif axis == 0:
            Pl = np.zeros((0, shp[1], shp[2]))
            Ph = np.zeros((0, shp[1], shp[2]))
        # todo:extend till end
        for i in range(shp[axis]):
            if axis == 1:
                
                # at the beginning
                if i < (window/2):
                    
                    subVl = self.Vl_1[:, :i+int(window/2), :]
                    subVh = self.Vh_1[:, :i+int(window/2), :]
            
                # at the end  
                elif i > (shp[axis] - 1) - (window/2):
                    subVl = self.Vl_1[:, i-int(window/2):, :]
                    subVh = self.Vh_1[:, i-int(window/2):, :]
                
                # otherwise
                else:
                    subVl = self.Vl_1[:, i-int(window/2):i+int(window/2), :]
                    subVh = self.Vh_1[:, i-int(window/2):i+int(window/2), :]
            
            # print(subVl.shape, subVh.shape)
            # elif axis == 0:
            #    subVl = self.Vl_1[i:i+window, :, :]
            #    subVh = self.Vh_1[i:i+window, :, :]
                
            #print(Pl.shape)
            #print(subVl.shape)
            
            maxVl = (np.amax(subVl, axis = axis, keepdims=True))
            
            #print(maxx.shape)
            
            #print('axis', axis)
            
            #print(Ph.shape)
            Pl = np.concatenate((Pl, maxVl), axis=axis)
            Ph = np.concatenate((Ph, np.amax(subVh, axis=axis, keepdims=True)), axis=axis)
            
        self.Vl_1 = Pl
        self.Vh_1 = Ph
        # print('shape after', self.Vl_1.shape, self.Vh_1.shape)
        
    def slidingMIP_old(self, axis=1):
        '''
        generate volume of mip's in a sliding fashion
        '''
        axis = int(axis)
        if axis > 2:
            axis = 2
        if axis < 0:
            axis = 0
        
        shp = self.Vl.shape
        
        window = 10
        
        if axis == 1:
            Pl = np.zeros((shp[0],0,shp[2]))
            Ph = np.zeros((shp[0],0,shp[2]))
        elif axis == 0:
            Pl = np.zeros((0, shp[1], shp[2]))
            Ph = np.zeros((0, shp[1], shp[2]))
                
        for i in range(shp[axis]-int(window)):
            if axis == 1:
                subVl = self.Vl[:, i:i+window, :]
                subVh = self.Vh[:, i:i+window, :]
            elif axis == 0:
                subVl = self.Vl[i:i+window, :, :]
                subVh = self.Vh[i:i+window, :, :]
                
            #print(Pl.shape)
            #print(subVl.shape)
            
            maxx = (np.amax(subVl, axis = axis, keepdims=True))
            
            #print(maxx.shape)
            
            #print('axis', axis)
            
            #print(Ph.shape)
            Pl = np.concatenate((maxx, Pl), axis=axis)
            Ph = np.concatenate((np.amax(subVh, axis=axis, keepdims=True), Ph), axis=axis)
            #Pl[i] = np.amax(subVl, axis = axis)
            #Ph[i] = np.amax(subVh, axis = axis)
        
        alpha = 16
        
        # RGB stack
        P = np.stack([Pl, alpha * Ph, np.zeros(Ph.shape)], axis = -1)
        
        P[P < 0] = 0
        P = exposure.rescale_intensity(P, out_range = np.uint8)
        P = P.astype(dtype=np.uint8)
        
        # rescale intensity
        val = np.quantile(P,(0.8, 0.9925))
        #print("Quantile", val[0], val[1], 'fixed values', round(0.03*255), 0.2*255, 'up to', 0.3*255)
        
        P = exposure.rescale_intensity(P, in_range = (val[0], val[1]), out_range = np.uint8)
        
        return P
        
    def calc_alpha(self, alpha):
        '''
        MIP helper function
        '''
        return np.sum(np.square(self.Pl - alpha * self.Ph))
    
    def calc_alpha_3d(self, alpha):
        '''
        MIP helper function
        '''
        return np.sum(np.square(self.Vl_split - alpha * self.Vh_split))
    
    def calcMIP3D(self, do_plot = True):
        '''
        plot maximum intensity projection along second axis
        options:
            do_plot = True
        '''
    
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
            
        # global alpha calculation (all slices)
        res = minimize_scalar(self.calc_alpha_3d, bounds=(0, 100), method='bounded')
        alpha = res.x
        
        #print(alpha)
        
        # RGB stack
        self.P_sliced = np.stack([self.Vl_split, alpha * self.Vh_split, np.zeros(self.Vl_split.shape)], axis = -1)
        
        self.P_sliced[self.P_sliced < 0] = 0
        self.P_sliced = exposure.rescale_intensity(self.P_sliced, out_range = np.uint8)
        self.P_sliced = self.P_sliced.astype(dtype=np.uint8)
        
        # rescale intensity
        val = np.quantile(self.P_sliced,(0.8, 0.9925))
        #print("Quantile", val[0], val[1], 'fixed values', round(0.03*255), 0.2*255, 'up to', 0.3*255)
        
        self.P_sliced = exposure.rescale_intensity(self.P_sliced, in_range = (val[0], val[1]), out_range = np.uint8)
    
        if do_plot:
            shp = self.P_sliced.shape
            P_ = self.P_sliced.copy()
            P_[:,:,-3:-1,:] = 255
            plt.figure()
            plt.imshow(P_.reshape((shp[0], shp[1]*shp[2], -1)))
            plt.title(str(self.file.ID))
            plt.show()
        
    def mergeVOLUME_RGB(self):
        '''
        merge low frequency and high frequency data feeding into different
        colour channels
        '''
        B = np.zeros(np.shape(self.Vl_1))
        
        self.Vm = np.stack([self.Vl_1, self.Vh_1, B], axis = -1)
        
    def cutDEPTH(self):
        ''' cut Vl and Vh to 500 x 171 x 333'''
        
        zmax = 500
        
        # extract shape
        shp = self.Vl.shape
        
        if shp[0] >= zmax:
            self.Vl = self.Vl[:500,:,:]
            self.Vh = self.Vh[:500,:,:]
        else:
            ext = zmax - shp[0]
            print('Extending volume. old shape:', shp)
            
            self.Vl = np.concatenate([self.Vl, np.zeros((ext, shp[1], shp[2]))], axis = 0)
            self.Vh = np.concatenate([self.Vh, np.zeros((ext, shp[1], shp[2]))], axis = 0)  
            
            print('New shape:', self.Vl.shape)

    def saveMIP(self, destination, fstr = ''):
        '''
        save MIP as 2d image
        '''
         # generate Path object
        destination = Path(destination)
        
        # generate filename
        img_file = (destination / ('R' + 
                                   self.file.DATETIME + 
                                   self.file.ID +
                                   '_' +
                                   fstr +
                                   '.png')).resolve()
        print(str(img_file))
        
        
        imageio.imwrite(img_file, self.P)

    def saveSURFACE(self, destination, fstr = ''):
        '''
        save surface as 2d image with colorbar?
        '''
        plt.ioff()
        plt.figure()
        
        Surf = self.Sip-np.amin(self.Sip)
        
        plt.imshow(Surf, cmap=cm.jet)
        plt.colorbar()
        #plt.show()
        
        
        
        
        # generate Path object
        destination = Path(destination)
        
        # generate filename
        img_file = (destination / ('R' + 
                                   self.file.DATETIME + 
                                   self.file.ID +
                                   '_' +
                                   fstr +
                                   '.png')).resolve()
        print(str(img_file))
        
        plt.title(str(self.file.ID))
        plt.savefig(img_file)
        
        plt.close() 
    
        
        #imageio.imwrite(img_file, self.Sip)
        
    def saveMIP3D(self, destination, fstr = ''):
        ''' save rgb maximum intensity projection volume'''
        
        # Vm is a 4-d numpy array, with the last dim holding RGB
        shape_3d = self.P_sliced.shape[0:3]
        rgb_dtype = np.dtype([('R', 'u1'), ('G', 'u1'), ('B', 'u1')])
        #self.Vm = self.P_sliced.astype('u1')
        self.P_sliced = self.P_sliced.copy().view(rgb_dtype).reshape(shape_3d)
        img = nib.Nifti1Image(self.P_sliced, np.eye(4))
        
        # generate Path object
        destination = Path(destination)
        
        # generate filename
        nii_file = (destination / ('R' + 
                                   self.file.DATETIME + 
                                   self.file.ID +
                                   '_' +
                                   fstr +
                                   '.nii.gz')).resolve()
        print(str(nii_file))
        nib.save(img, str(nii_file))
        
    def saveVOLUME(self, destination, fstr = ''):
        '''
        save rgb volume
        '''
        
        self.Vm = exposure.rescale_intensity(self.Vm, out_range = np.uint8)
        
        # Vm is a 4-d numpy array, with the last dim holding RGB
        shape_3d = self.Vm.shape[0:3]
        rgb_dtype = np.dtype([('R', 'u1'), ('G', 'u1'), ('B', 'u1')])
        self.Vm = self.Vm.astype('u1')
        self.Vm = self.Vm.copy().view(rgb_dtype).reshape(shape_3d)
        img = nib.Nifti1Image(self.Vm, np.eye(4))
        
        
        # generate Path object
        destination = Path(destination)
        
        # if this is a cut file, need to construct the cut z-value
        # this is only for vesnet preparation
        if self.layer_end is not None:
            z_cut = '_' + 'z' + str(self.layer_end)
        else:
            z_cut = ''
        
        # generate filename
        nii_file = (destination / ('R' + 
                                   self.file.DATETIME + 
                                   self.file.ID +
                                   z_cut +
                                   '_' +                                   
                                   fstr +
                                   '.nii.gz')).resolve()
        print(str(nii_file))

        
        nib.save(img, str(nii_file))
        
    class FileStruct():
        '''
        helper class for data management
        '''
        def __init__(self, filepathLF, filepathHF, filepathSURF, ID, DATETIME):
            self.LF = filepathLF
            self.HF = filepathHF
            self.SURF = filepathSURF
            self.ID = ID
            self.DATETIME = DATETIME
    
    @staticmethod
    def loadNII(path):
        img = nib.load(path)
        return img.get_fdata()
 

class RSOM_vessel(RSOM):
    '''
    additional methods for preparing RSOM data for vessel segmentation,
    e.g. cut away epidermis
    '''
    
    def cutLAYER(self, path, mode='pred', fstr='layer_pred.nii.gz'):
        '''
        cut off the epidermis with loading corresponding segmentation mask.
        '''
        print('cutLayer method')
        
        # generate path
        filename = 'R' + self.file.DATETIME + self.file.ID + '_' + fstr
        file = os.path.join(path, filename)
        
        print('Loading', file)
        
        # two modes supported, extract from prediction volume
        # or manual input through file   
        if mode == 'pred':
            
            img = nib.load(file)
            self.S = img.get_fdata()
            self.S = self.S.astype(np.uint8)
            
            assert self.Vl.shape == self.S.shape, 'Shapes of raw and segmentation do not match'
            
            print(self.Vl.shape)
            print(self.S.shape)
            
            
            # for every slice in x-y plane, calculate label sum
            label_sum = np.sum(self.S, axis=(1, 2))
            
            max_occupation = np.amax(label_sum) / (self.S.shape[1] * self.S.shape[2])
            max_occupation_idx = np.argmax(label_sum)
            
            print('Max occ', max_occupation)
            print('idx max occ', max_occupation_idx)
            
            # normalize
            label_sum = label_sum.astype(np.double) / np.amax(label_sum)
            
            # define cutoff parameter
            cutoff = 0.05
            
            label_sum_bin = label_sum > cutoff
                 
            label_sum_idx = np.squeeze(np.nonzero(label_sum_bin))
        
            layer_end = label_sum_idx[-1]
            
            # additional fixed pixel offset
            offs = 5
            layer_end += offs
    
        elif mode == 'manual':
            f = open(file)
            layer_end = int(str(f.read()))
        else:
            raise NotImplementedError
            
        
        print('Cutting at', layer_end)
        # replace with zeros
        #self.Vl[:layer_end,:,:] = 0
        #self.Vh[:layer_end,:,:] = 0
        
        # cut away
        self.Vl = self.Vl[layer_end:, :, :]
        self.Vh = self.Vh[layer_end:, :, :]
        
        self.layer_end = layer_end

        
        
        # keep meta information?
        
        
        
    def rescaleINTENSITY(self):
        '''
        overrides method in class RSOM, because vessel segmentation needs 
        different rescalé
        '''
        print('Vessel rescaleINTENSITY method')
        self.Vl_1 = exposure.rescale_intensity(self.Vl_1, in_range = (0, 0.25))
        self.Vh_1 = exposure.rescale_intensity(self.Vh_1, in_range = (0, 0.15))
        #self.Vl_1[:,:,:] = 0
            
        self.Vl_1 = self.Vl_1**2
        self.Vh_1 = self.Vh_1**2
            
        self.Vl_1 = exposure.rescale_intensity(self.Vl_1, in_range = (0.05, 1))
        self.Vh_1 = exposure.rescale_intensity(self.Vh_1, in_range = (0.05, 1))
        
        
    def thresholdSEGMENTATION(self):
        
        self.Vseg = np.logical_or(np.logical_or((self.Vh_1 + self.Vl_1) >= 1, 
                                                self.Vl_1 > 0.3),
                                    self.Vh_1 > 0.7)
        return self.Vseg
        
    def mathMORPH(self):
        
        # TODO PADDING AND REMOVE 
        # probably the problem is in binary_closing??

        
        filter = ndimage.morphology.generate_binary_structure(3,1).astype(np.int64)
        print(filter)
        M = ndimage.convolve(self.Vseg.astype(np.int64), filter, mode='constant',cval=0)
        
        # fill holes custom function
        holesMask = np.logical_or(M == 6, M == 5)
        self.Vseg = np.logical_or(self.Vseg, holesMask)
        
        # remove single pixels or only 2 pixels
        singleMask = M >= 2
        self.Vseg = np.logical_and(singleMask, self.Vseg)

        # closing
        self.Vseg = (ndimage.morphology.binary_closing(
                np.pad(self.Vseg, 1, mode='edge')))[1:-1, 1:-1, 1:-1]
        
        # remove small objects
        # 30 seems a good value, tested from 10 to 40
        self.Vseg = morphology.remove_small_objects(self.Vseg, 30)
        
        
        
        
        
    def saveSEGMENTATION(self, destination, fstr='th'):
        '''
        save rgb volume
        '''
    
        Vseg = self.Vseg.astype(np.uint8)
        img = nib.Nifti1Image(Vseg, np.eye(4))
        
        path = os.path.join(destination,
                            'R' + 
                            self.file.DATETIME + 
                            self.file.ID +
                            '_' +                                
                            fstr +
                            '.nii.gz')
                            
        
        nib.save(img, path)
        
        
    def saveVOLUME_float(self, destination, fstr = ''):
        '''
        override method from RSOM class
        save rgb volume
        '''
        
        #self.Vm = exposure.rescale_intensity(self.Vm, out_range = np.uint8)
        
        # Vm is a 4-d numpy array, with the last dim holding RGB
        #dtype_ = 'u1'
        #dtype_ = 'f4'  # np.float32
        
        #shape_3d = self.Vm.shape[0:3]
        #rgb_dtype = np.dtype([('R','f4'), ('G', 'f4'), ('B', 'f4')])
        self.Vm = self.Vm.astype(np.float32)
        
        #self.Vm = self.Vm.copy().view(rgb_dtype).reshape(shape_3d)
        
        V_R = self.Vm[...,0]
        V_G = self.Vm[...,1]
        
        
        for V, c in [(V_R, 'R'), (V_G, 'G')]:
        
            img = nib.Nifti1Image(V, np.eye(4))
            
            
            # generate Path object
            destination = Path(destination)
            
            # if this is a cut file, need to construct the cut z-value
            # this is only for vesnet preparation
            if self.layer_end is not None:
                z_cut = '_' + 'z' + str(self.layer_end)
            else:
                z_cut = ''
            
            # generate filename
            nii_file = (destination / ('R' + 
                                       self.file.DATETIME + 
                                       self.file.ID +
                                       z_cut +
                                       '_' +                                   
                                       fstr +
                                       '_' +
                                       c +
                                       '.nii.gz')).resolve()
            print(str(nii_file))
        
            
            nib.save(img, str(nii_file))
        
            
class RSOM_mip_interp():
    def __init__(self, filepath):
        ''' create an instance of RSOM_MIP_IP; requires path '''
        self.filepath = filepath
        
    def readNII(self):
        ''' read in the .nii.gz files specified in self.filepath'''
        
        img = nib.load(str(self.filepath))
        self.L_sliced = img.get_fdata()
        self.L_sliced = self.L_sliced.astype(np.uint8)
        
    def saveNII(self, destination, fstr = ''):
        ''' '''
        

        self.L = self.L.astype(np.uint8)
        img = nib.Nifti1Image(self.L, np.eye(4))
        
        file = self.filepath
        
        name = file.name
        
        name = name.replace('mip3d_l.nii.gz',('l' + fstr + '.nii.gz'))
        
            # generate Path object
        destination = Path(destination)
        
        # generate filename
        nii_file = destination / name
        print(str(nii_file))
        nib.save(img, str(nii_file))
        
    def interpolate(self):
        ''' interpolate the .nii.gz label files along axis=1 '''
        
        # label volume is           depth x 9 x 333
        # need to interpolate to    depth x 171 x 333
        
        x_mip = 9
        x_rep = 171
        
        label_min = np.amin(self.L_sliced)
        label_max = np.amax(self.L_sliced)
        
        #self.L_sliced = exposure.rescale_intensity(self.L_sliced, in_range = (label_min, label_max), out_range = np.uint8)
        #self.L_sliced = self.L_sliced.astype(np.uint8)
        
        shp = self.L_sliced.shape
    
        # create grid for L_sliced
        x1 = np.arange(shp[0])
        x2 = np.linspace(np.floor(x_rep/x_mip/2), x_rep - np.ceil(x_rep/x_mip/2), num = x_mip)
        x3 = np.arange(shp[2])
        
        # dimension to be interpolated
        x2_q = np.arange(x_rep)
        
        
        #x1_q, x2_q, x3_q = (
                #x1[:, None, None], x2_q[None, :, None], x3[None, None, :])
        
        # TODO: 
        #Interpolation somehow still does not work
        # another approach, along axis=0, search for change in value,
        # and create a 2d function, based on the indices
        
        # it works, just need to stack a for loop on top, in order to 
        # process an aribitrary number of layers in z-direction
        
        # in case, label order is random, sort
        
        # determine label order of input
        
        # create synthetic extra label
        # TODO: REMOVE THIS
        #self.L_sliced[0:20,:,:] = 2
        
        L_sliced_ = self.L_sliced.copy().astype(np.int64)
        
        # cut in direction dim 0 throught middle of the volume
        L_1d = L_sliced_[:,int(shp[1]/2),int(shp[2]/2)].copy()
        
        # find the number of labels, and their indices
        # np.unique returns the labels in ascending order
        labels, idx = np.unique(L_1d, return_index = True)
        
        n_labels = labels.size
        
        # TODO:
        print('Warning. Processing data with poor implementation for label handling')
        
        for xx in np.arange(x_mip):
            for yy in np.arange(shp[2]):
            
               idx_nz = np.nonzero(L_sliced_[:, xx, yy])
               #print(idx_nz)
               idx_nz = idx_nz[0][-1]
               #print(idx_nz)
               L_sliced_[idx_nz+1:, xx, yy] = 2
               
               #print(np.unique(L_sliced_[:, xx, yy], return_index = True))
               
               #print(L_sliced_[idx_nz, xx, yy])
               
        # TODO: 
        #DANGEROUS HACK
        print('warning, remove that dangerous hack')
        n_labels = n_labels + 1
        
        
        
        # check if labels are in shape: 0 1 2 3 4 already
        # TODO: check boolean expression
#        if not ((labels[0] == 0) and (np.any(np.diff(labels) - 1))):
#            
#            print('reshaping')
#            # if not: reshape
#            # add some 'large' number to the labels
#            L_sliced_ += 20
#            
#            layer_ctr = 0
#            
#            for nl in np.arange(n_labels):
#                L_sliced_[L_sliced_ == labels[nl] + 20] = layer_ctr
#                layer_ctr += 1
                
        # INPUT: in dim 0: ascending index: label order: 0 1 2 3 4
        self.L = np.zeros((shp[0], x_rep, shp[2]))
        
        for nl in np.arange(n_labels - 1):
            surf = np.zeros((x_mip, shp[2]))
    
            for xx in np.arange(x_mip):
                for yy in np.arange(shp[2]):
                    
                    #idx = np.nonzero(np.logical_not(self.L_sliced[:, xx, yy]))
                    idx = np.nonzero(L_sliced_[:, xx, yy])
                    
                    surf[xx, yy] = idx[0][0]
                    
            fn = interpolate.interp2d(x3, x2, surf, kind = 'cubic')
            surf_ip = fn(x3, x2_q)
            
            
            
            for xx in np.arange(x_rep):
                for yy in np.arange(shp[2]):
                    self.L[0:np.round(surf_ip[xx, yy]).astype(np.int), xx, yy] += 1
            
            # NEXT LABEL
            L_sliced_ -= 1
            L_sliced_[L_sliced_ < 0] = 0
            
        # TODO   
        print('Warning. Processing data with poor implementation for label handling')
        self.L[self.L == 2] = 0
        
            
        return self.L
    
        # RESULT: in dim 0: ascending index: label order: 4 3 2 1 0
        
        
#        self.L = interpolate.interpn(
#                (x1, x2, x3),
#                self.L_sliced,
#                (x1_q, x2_q, x3_q),
#                method='linear',
#                bounds_error = False,
#                fill_value = None)
#        
#        self.L_255 = self.L.copy()
#        
#        self.L = np.round(self.L)
#        self.L = exposure.rescale_intensity(self.L, in_range = np.uint8, out_range = (label_min, label_max))
#        
#        # TODO EROSION DILATION?
#        #scipy.ndimage.morphology.grey_closing
#        
#        print('Shape of L', self.L.shape)

