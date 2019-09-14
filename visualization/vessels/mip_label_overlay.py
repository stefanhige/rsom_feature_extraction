#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 14 17:06:11 2019

@author: stefan
"""

import imageio
import numpy as np
import nibabel as nib
import os
from skimage import filters


mip_path = '/home/stefan/PYTHON/HQDatasetVesselAnnot/out_from_prep/'
in_filename = 'R_20190605163439_HQ0003_mip.png'

mip_path = os.path.join(mip_path, in_filename)

label_path = '/home/stefan/PYTHON/HQDatasetVesselAnnot/vessels/R_20190605163439_HQ0003_th_corrected_rso.nii.gz'



# load image
mip = imageio.imread(mip_path)


# load label
label = (nib.load(label_path)).get_fdata()

z0 = 98

mip_label = np.sum(label, axis=1) >= 1

mip_label = np.concatenate((np.zeros((z0, mip_label.shape[-1]), dtype=np.bool), 
                            mip_label.astype(np.bool)), axis = 0)



out_path = '/home/stefan/PYTHON/HQDatasetVesselAnnot/test_mip_overlay'
out_filename = in_filename.replace('.png', '_ol___.png')
out_path = os.path.join(out_path, out_filename)


mip_label = mip_label.astype(np.float32)
mip_label_edge = filters.sobel(mip_label)
mip_label_edge = mip_label_edge/np.amax(mip_label_edge)

# feed into blue channel
blue = 150*mip_label + 200*mip_label_edge
blue[blue>255] = 255
blue = blue.astype(np.uint8)

mip_overlay = mip.copy()
mip_overlay[:, :, 2] = blue

imageio.imwrite(out_path, mip_overlay)
