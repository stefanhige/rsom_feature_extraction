#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 15:45:39 2019

@author: stefan
"""
import numpy as np
import os
import nibabel as nib
import shutil

from dataloader import RSOMVesselDataset
from dataloader import DropBlue, ToTensor, to_numpy
from patch_handling import get_volume

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils



def saveNII(V, path):
        img = nib.Nifti1Image(V, np.eye(4))
        nib.save(img, str(path))
        

def test_dl(result):

    # 1. generate random data and label files
    L_dim = D_dim = (100, 100, 100)
    
    D = [None, None, None]
    L = [None, None, None]
    Dname = [None, None, None]
    Lname = [None, None, None]
    
    D[0] = np.random.random_sample(D_dim)
    D[1] = np.random.random_sample(D_dim)
    D[2] = np.random.random_sample(D_dim)
    
    L[0] = np.random.random_sample(L_dim)
    L[1] = np.random.random_sample(L_dim)
    L[2] = np.random.random_sample(L_dim)
    
    D[0] = D[0].astype(dtype=np.float32)
    D[1] = D[1].astype(dtype=np.float32)
    D[2] = D[2].astype(dtype=np.float32)
    
    L[0] = L[0].astype(dtype=np.float32)
    L[1] = L[1].astype(dtype=np.float32)
    L[2] = L[2].astype(dtype=np.float32)
    
    # 2. generate test directory
    cwd = os.getcwd()
    testdir = os.path.join(cwd,'temp_test_dl')
    if os.path.exists(testdir):
        shutil.rmtree(testdir)
    os.mkdir(testdir)
    
    # 3. save files to test directory
    Dname[0] = '1_v_rgb.nii.gz'
    Dname[1] = '2_v_rgb.nii.gz'
    Dname[2] = '3_v_rgb.nii.gz'
    
    Lname[0] = '1_v_l.nii.gz'
    Lname[1] = '2_v_l.nii.gz'
    Lname[2] = '3_v_l.nii.gz'
    
    saveNII(D[0], os.path.join(testdir, Dname[0]))
    saveNII(D[1], os.path.join(testdir, Dname[1]))
    saveNII(D[2], os.path.join(testdir, Dname[2]))
    
    saveNII(L[0], os.path.join(testdir, Lname[0]))
    saveNII(L[1], os.path.join(testdir, Lname[1]))
    saveNII(L[2], os.path.join(testdir, Lname[2]))
    
    # 4. construct dataset and dataloader
    
    divs = (2, 3, 5)
    offset = (0, 0, 0)
     
    # TODO transforms
    set1 = RSOMVesselDataset(testdir, 
                             divs=divs, 
                             offset = offset,
                             transform=transforms.Compose([ToTensor()]))
    
    dataloader = DataLoader(set1,
                            batch_size=1, 
                            shuffle=False, 
                            num_workers=1, 
                            pin_memory=True)
    
    
    
    # 5. draw samples and reconstruct the patches to volumes.
    #try:
    if 1:
        set1_iter = iter(dataloader)
        rem = []
        
        # 3 files
        for file in np.arange(len(Dname)):
            Dout = []
            Lout = []
            
            # prod(divs) patches in each sample
            for ctr in np.arange(np.prod(divs)):
                patch = next(set1_iter)
                #print('INDEX:', patch['meta']['index'])
                #print('filename:',patch['meta']['filename'])
                #print(patch['data'].shape)
                Dout.append(patch['data'].squeeze())
                Lout.append(patch['label'].squeeze())
            
            if isinstance(patch['data'], torch.Tensor):
                Dout = (torch.stack(Dout)).numpy()
                Lout = (torch.stack(Lout)).numpy()
            else:
                Dout = np.array(Dout)
                Lout = np.array(Lout)
            
            print('Dout, Lout shapes:', Dout.shape, Lout.shape)
            Dvol_ = get_volume(Dout, divs, offset)
            Lvol_ = get_volume(Lout, divs, offset)
                
            print('Dvol, Lvol shapes:', Dvol_.shape, Lvol_.shape)
            
            Dvol = to_numpy(Dvol_, 
                            patch['meta'],
                            Vtype='data',
                            dimorder='torch')
            Lvol = to_numpy(Lvol_, 
                            patch['meta'],
                            Vtype='label',
                            dimorder='torch')
            
            rem.append(Dvol==D[Dname.index(patch['meta']['filename'][0])])
            
            # 6. compare to generated data.
            
            Dbool = Dvol == D[Dname.index(patch['meta']['filename'][0])]
            Lbool = Lvol == L[Dname.index(patch['meta']['filename'][0])]
            
            # apply crop what was reconstructed with zeros
            b = patch['meta']['dcrop']['begin'].numpy()[0]
            e = patch['meta']['dcrop']['end'].numpy()[0]
            if e[0] == 0:
                Dbool = Dbool[b[0]:,...]
                Lbool = Lbool[b[0]:,...]
            else:
                Dbool = Dbool[b[0]:e[0],...]
                Lbool = Lbool[b[0]:e[0],...]
            if e[1] == 0:
                Dbool = Dbool[:, b[0]:,...]
                Lbool = Lbool[:, b[0]:,...]
    
            else:
                Dbool = Dbool[:, b[0]:e[0],...]  
                Lbool = Lbool[:, b[0]:e[0],...]              
            if e[2] == 0:
                Dbool = Dbool[...,b[0]:]
                Lbool = Lbool[...,b[0]:]
            else:
                Dbool = Dbool[...,b[0]:e[0]]  
                Lbool = Lbool[...,b[0]:e[0]]          
            if np.all(Dbool) and np.all(Lbool):
                print('Test passed')
                result.append(0)
            else:
                print('Test failed')
                result.append(1)
    

    
    #except:
        #print('ALL TEST CODE FAILED')
   
    # 7 delete test directory
        
    #print('delete dir')
    shutil.rmtree(testdir)
    
    return result


results = []
test_dl(results)

print('RESULTS:', results)
        
