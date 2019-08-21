#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 21 16:41:18 2019

@author: stefan
"""
import numpy as np

from patch_handling import get_patch_ndim


# TEST 1D
A = np.random.random_sample((100))
patches = []
for idx in np.arange(2):
    print(idx)
    
    A_1patch_new = get_patch_ndim(A, idx, divs=2, offset=6)
       
    patches.append(A_1patch_new)
    
patches = np.array(patches)

stop

# TEST 2D

A = np.random.random_sample((100, 100))


patches = []
for idx in np.arange(4):
    print(idx)
    
    A_1patch_new = get_patch_ndim(A, idx, divs=(2,2), offset=(6,6))
       
    patches.append(A_1patch_new)
    
patches = np.array(patches)
