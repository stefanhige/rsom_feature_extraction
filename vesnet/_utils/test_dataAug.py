
import torch
import numpy as np

import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

import nibabel as nib
import warnings


from dataloader import RSOMVesselDataset, DataAugmentation


# test data augmentaion



# load one sample 

dir = '/home/gerlstefan/data/vesnet/annotatedDataset/eval'

train_dataset = RSOMVesselDataset(dir,
                                  divs=(1, 1, 1), 
                                  offset=(0, 0, 0),
                                  transform=DataAugmentation())

# apply 10 times random data augmentation
for i in range(5):
    sample = train_dataset[0]
    data = sample['data']
    label = sample['label']

    print(data.shape)
    print(label.shape)
 
    shape_3d = data.shape[0:3]
    rgb_dtype = np.dtype([('R', 'u1'), ('G', 'u1'), ('B', 'u1')])
    data_nii = data.astype('u1')
    data_nii = data_nii.copy().view(rgb_dtype).reshape(shape_3d)
    img = nib.Nifti1Image(data_nii, np.eye(4))
    nib.save(img, 'aug/aug{:d}.nii.gz'.format(i))

    # save segmentation
    img = nib.Nifti1Image(label.astype(np.uint8), np.eye(4))
    nib.save(img, 'aug/aug_l_{:d}.nii.gz'.format(i))
 
    
    



# save all 10 and look if it looks valid
