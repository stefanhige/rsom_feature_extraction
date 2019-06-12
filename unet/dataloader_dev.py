#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 10 14:48:49 2019

dataloader test script

@author: sgerl
"""

import torch
import numpy as np
import matplotlib.pyplot as plt

#from skimage import color

import os

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

import nibabel as nib


class RSOMLayerDataset(Dataset):
    """rsom dataset class for layer segmentation"""

    def __init__(self, 
                 root_dir, 
                 data_str='_rgb.nii.gz', 
                 label_str='_l.nii.gz', 
                 transform=None):
        """
        Args:
            root_dir (string): Directory with all the nii.gz files.
            data_str (string): end part of filename of training data.
            label_str (string): end part of filename of segmentation ground truth data.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        assert os.path.exists(root_dir) and os.path.isdir(root_dir), \
        'root_dir not a valid directory'
        
        self.root_dir = root_dir
        self.transform = transform
        
        assert isinstance(data_str, str) and isinstance(label_str, str), \
        'data_str or label_str not valid.'
        
        self.data_str = data_str
        self.label_str = label_str
        
        # get all files in root_dir
        all_files = os.listdir(path = root_dir)
        # extract the  data files
        self.data = [el for el in all_files if el[-len(data_str):] == data_str]
        
        assert len(self.data) == \
            len([el for el in all_files if el[-len(label_str):] == label_str]), \
            'Amount of data and label files not equal.'

    def __len__(self):
        return len(self.data)
    
    @staticmethod
    def _readNII(rpath):
        '''
        read in the .nii.gz file
        Args:
            rpath (string)
        '''
        
        img = nib.load(str(rpath))
        
        # TODO: when does nib get_fdata() support rgb?
        # currently not, need to use old method get_data()
        return img.get_data()

    def __getitem__(self, idx):
        data_path = os.path.join(self.root_dir, 
                            self.data[idx])
        label_path = os.path.join(self.root_dir, 
                                   self.data[idx].replace(self.data_str, self.label_str))
        
        # read data
        data = self._readNII(data_path)
        data = np.stack([data['R'], data['G'], data['B']], axis=-1)
        data = data.astype(np.float32)
        
        # read label
        label = self._readNII(label_path)
        label = label.astype(np.float32)
        
        # add meta information
        meta = {'filename': self.data[idx],
                'dcrop':{'begin': None, 'end': None},
                'lcrop':{'begin': None, 'end': None}}

        sample = {'data': data, 'label': label, 'meta': meta}

        if self.transform:
            sample = self.transform(sample)

        return sample
    


# transform
class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        data, label, meta = sample['data'], sample['label'], sample['meta']
        
        # data is [Z x X x Y x 3] [500 x 171 x 333 x 3]
        # label is [Z x X x Y] [500 x 171 x 333]
        
        # we want one sample to be [Z x Y x 3]  2D rgb image
        
        # numpy array size of images
        # [H x W x C]
        # torch tensor size of images
        # [C x H x W]
        
        # and for batches
        # [B x C x H x W]
        
        # here, X is the batch size.
        # so we want to reshape to
        # [X x C x Z x Y] [171 x 3 x 500 x 333]
        data = data.transpose((1, 3, 0, 2))
        
        # and for the label
        # [X x Z x Y] [171 x 500 x 333]
        label = label.transpose((1, 0, 2))
        

        return {'data': torch.from_numpy(data),
                'label': torch.from_numpy(label),
                'meta': meta}
        

# class random z-shift      
class RandomZShift(object):
    """Apply random z-shift to sample.

    Args:
        max_shift (int, tuple of int):  maximum acceptable 
                                        shift in -z and +z direction (in voxel)
        
    """

    def __init__(self, max_shift=0):
        assert isinstance(max_shift, (int, tuple))
        if isinstance(max_shift, int):
            self.max_shift = (-max_shift, max_shift)
        else:
            assert len(max_shift) == 2
            assert max_shift[1] > max_shift[0]
            self.max_shift = max_shift

    def __call__(self, sample):
        data, label, meta = sample['data'], sample['label'], sample['meta']
        assert isinstance(data, np.ndarray)
        assert isinstance(label, np.ndarray)
        
        # initial shape
        data_ishape = data.shape
        label_ishape = label.shape
        
        # generate random dz offset
        dz = int(round((self.max_shift[1] - self.max_shift[0]) * np.random.random_sample() + self.max_shift[0]))
        assert (dz >= self.max_shift[0] and dz <= self.max_shift[1])
        
        if dz:
            shift_data = np.zeros(((abs(dz), ) + data.shape[1:]), dtype = np.uint8)
            shift_label = np.zeros(((abs(dz), ) + label.shape[1:]), dtype = np.uint8)
        
            print('RandomZShift: Check if this array modification does the correct thing before actually using it')
            # TODO: verify
            data = np.concatenate((data[:-abs(dz),:,:,:], shift_data)\
                    if dz > 0 else (shift_data, data[abs(dz):,:,:,:]), axis = 0)
            label = np.concatenate((label[:-abs(dz),:,:], shift_label)\
                    if dz > 0 else (shift_label, label[abs(dz):,:,:]), axis = 0)

            # should be the same...
            assert (data_ishape == data.shape and label_ishape == label.shape)
        
        return {'data': data, 'label': label, 'meta': meta}
    
# class normalize
class ZeroCenter(object):
    """ 
    Zero center input volumes
    """    
    def __call__(self, sample):
        data, label, meta = sample['data'], sample['label'], sample['meta']
        assert isinstance(data, np.ndarray)
        assert isinstance(label, np.ndarray)
        
        # compute for all x,y,z mean for every color channel
        rgb_mean = np.around(np.mean(data, axis=(0, 1, 2))).astype(np.int16)
        meanvec = np.tile(rgb_mean, (data.shape[:-1] + (1,)))
        
        data -= meanvec
        
        return {'data': data, 'label': label, 'meta': meta}
    
 
# TODO CLASS implementation
class CropToEven(object):
    """ 
    if Volume shape is not even numbers, simply crop the first element
    except for last dimension, this is RGB  = 3
    """
    def __call__(self, sample):
        data, label, meta = sample['data'], sample['label'], sample['meta']
        assert isinstance(data, np.ndarray)
        assert isinstance(label, np.ndarray)
     
        IsOdd = np.mod(data.shape[:-1], 2)
        
        data = data[IsOdd[0]:, IsOdd[1]:, IsOdd[2]:, : ]
        label = label[IsOdd[0]:, IsOdd[1]:, IsOdd[2]:]
        
        # save, how much data was cropped
        # using torch tensor, because dataloader will convert anyways
        meta['dcrop']['begin'] = torch.from_numpy(np.array([IsOdd[0], IsOdd[1], IsOdd[2], 0], dtype=np.int16))
        meta['dcrop']['end'] = torch.from_numpy(np.array([0, 0, 0, 0], dtype=np.int16))
            
        meta['lcrop']['begin'] = torch.from_numpy(np.array([IsOdd[0], IsOdd[1], IsOdd[2]], dtype=np.int16))
        meta['lcrop']['end'] = torch.from_numpy(np.array([0, 0, 0], dtype=np.int16))
        
        return {'data': data, 'label': label, 'meta': meta}
    
    
    
    
    
#def plotMIP(sample, pred=None):
#    '''
#    result visualization
#    '''
#    if isinstance(sample, dict):
#        data, label = sample['data'].numpy(), sample['label'].numpy()
##        nInfo = 1
##    else:
##        data = sample
##        if pred:
##            nInfo = 2
##        else:
##            nInfo = 1
#    #TODO: this can do better.
#    # new approach: MIP is greyscale
#    # missing prediction visualization
#    # maybe display MIP in black and white
#    
#    # maximum intensity projection
#    P = np.amax(data, axis=0)
#    P = np.moveaxis(P, 0, 2)
#    P[:,:,1] = 0.8*P[:,:,1] 
#    P[P<0] = 0
#    Pgray = color.rgb2gray(P)
#    print(Pgray.shape)
#    
#    
#    # replace amax with if sum / length >= 0.5
#    # i.e. more than half the values are 1
#    L_masked = np.amax(label, axis=0)
#
#    print(L_masked.shape)
#
#
#    
#    plt.figure()
#    
#    plt.imshow(Pgray, cmap='gray')
#    
#    #Pl_masked = np.dstack([Pl_masked, Pl_masked, Pl_masked])
#    Pl_masked = np.ma.masked_where(Pl_masked==0, Pl_masked)
#    
#    #plt.imshow(Pl.astype(np.uint8), alpha=0.5
#    plt.pcolormesh(Pl_masked, facecolor='m')
#    
#    plt.figure()
#    plt.imshow(P.astype(np.uint8))
#    
#
#
#    return P


#import numpy as np              #Used for holding and manipulating data
#import numpy.random             #Used to generate random data
#import matplotlib as mpl        #Used for controlling color
#import matplotlib.colors        #Used for controlling color as well
#import matplotlib.pyplot as plt #Use for plotting
#
##Generate random data
#a = np.random.random(size=(10,10))
#
##This 30% of the data will be red
#am1 = a<0.3                                 #Find data to colour special
#am1 = np.ma.masked_where(am1 == False, am1) #Mask the data we are not colouring
#
##This 10% of the data will be green
#am2 = np.logical_and(a>=0.3,a<0.4)          #Find data to colour special
#am2 = np.ma.masked_where(am2 == False, am2) #Mask the data we are not colouring
#
##Colourmaps for each special colour to place. The left-hand colour (black) is
##not used because all black pixels are masked. The right-hand colour (red or
##green) is used because it represents the highest z-value of the mask matrices
#cm1 = mpl.colors.ListedColormap(['black','red'])
#cm2 = mpl.colors.ListedColormap(['black','green'])
#
#fig = plt.figure()                          #Make a new figure
#ax = fig.add_subplot(111)                   #Add subplot to that figure, get ax
#
##Plot the original data. We'll overlay the specially-coloured data
#ax.imshow(a,   aspect='auto', cmap='Greys', vmin=0, vmax=1)
#
##Plot the first mask. Values we wanted to colour (`a<0.3`) are masked, so they
##do not show up. The values that do show up are coloured using the `cm1` colour
##map. Since the range is constrained to `vmin=0, vmax=1` and a value of
##`cm2==True` corresponds to a 1, the top value of `cm1` is applied to all such
##pixels, thereby colouring them red.
#ax.imshow(am1, aspect='auto', cmap=cm1, vmin=0, vmax=1);
#ax.imshow(am2, aspect='auto', cmap=cm2, vmin=0, vmax=1);
#plt.show()




    #plt.title(str(self.file.ID))
    #plt.imshow(P, aspect = 1/4)
    #plt.show()
    #self.Pl = np.amax(self.Vl, axis = axis)
    #self.Ph = np.amax(self.Vh, axis = axis)
        
    # calculate alpha
    #res = minimize_scalar(self.calc_alpha, bounds=(0, 100), method='bounded')
    #alpha = res.x
        
    #self.P = np.dstack([self.Pl, alpha * self.Ph, np.zeros(self.Ph.shape)])
        
    # cut negative values, in order to allow rescale to uint8
    #self.P[self.P < 0] = 0
        
    #self.P = exposure.rescale_intensity(self.P, out_range = np.uint8)
    #self.P = self.P.astype(dtype=np.uint8)
    
    
    
#def plotMIP_sliced():
#    '''
#    
#    '''
#    pass


#def colorEncode(labelmap, colors, mode='RBG'):
#    labelmap = labelmap.astype(np.uint8)
#    labelmap_rgb = np.zeros((labelmap.shape[0], labelmap.shape[1], 3),
#                            dtype=np.uint8)
#    for label in np.unique(labelmap):
#        if label < 0:
#            continue
#        labelmap_rgb += (labelmap == label)[:, :, np.newaxis] * \
#            np.tile(colors,
#                    (labelmap.shape[0], labelmap.shape[1], 1))
#
#    if mode == 'BGR':
#        return labelmap_rgb[:, :, ::-1]
#    else:
#        return labelmap_rgb
#    
#    
#def visualize_result(data, pred, args):
#    (img, seg, info) = data
#
#    # segmentation
#    seg_color = colorEncode(seg, colors)
#
#    # prediction
#    pred_color = colorEncode(pred, colors)
#
#    # aggregate images and save
#    im_vis = np.concatenate((img, seg_color, pred_color),
#                            axis=1).astype(np.uint8)
#
#    img_name = info.split('/')[-1]
#    cv2.imwrite(os.path.join(args.result,
#                img_name.replace('.jpg', '.png')), im_vis)
            
        
        
    
        
        
# TODO:
# normalization?
# zero centering?
# torch tensor image visualization+
# of mip

        


# ==============================TEST===========================================

# root_dir = '/home/sgerl/Documents/PYTHON/TestDataset20190411/selection/layerseg/dataloader_dev'

# Obj = RSOMLayerDataset(root_dir, transform=transforms.Compose([RandomZShift(), ZeroCenter(), ToTensor()]))

# sample = Obj[0]

# P = plotMIP(sample)

#Obj = RSOMLayerDataset(root_dir, transform=transforms.Compose([RandomZShift(), ZeroCenter(), ToTensor()]))
#dat = Obj.readNII('/home/sgerl/Documents/PYTHON/TestDataset20190411/selection/layerseg/dataloader_dev/R_20170724150057_PAT001_RL01_l.nii.gz')

#sample = Obj[0]

#dataloader = DataLoader(Obj, batch_size=1, shuffle=False, num_workers=0)

# get a sample
#sample_dl = next(iter(dataloader))

#for sample in dataloader:
    #print(sample['data'].shape)

# numpy array size of images
# [H x W x C]
# torch tensor size of images
# [C x H x W]

# and for batches
# [B x C x H x W]

#np4d = sample['data']
#shape_3d = np4d.shape[0:3]
#rgb_dtype = np.dtype([('R', 'u1'), ('G', 'u1'), ('B', 'u1')])
#np4d = np4d.copy().view(rgb_dtype).reshape(shape_3d)
#img = nib.Nifti1Image(np4d, np.eye(4))
#       
#nib.save(img, '/home/sgerl/Documents/PYTHON/TestDataset20190411/selection/layerseg/dataloader_dev/out.nii.gz')
