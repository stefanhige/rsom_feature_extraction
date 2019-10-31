# shuffle individual slices of training and validation set, and then build one mega .nii stacked file.


import os
import random
import shutil
import nibabel as nib
import numpy as np

# training set size 20, val set size 5

# original dataset is in layer_seg

def _readNII(path):
    img = nib.load(str(path))
    return img.get_data()

def readData(path):
    data = _readNII(path)
    return np.stack([data['R'], data['G'], data['B']], axis=-1).astype(np.uint8)

def readLabel(path):
    return _readNII(path).astype(np.uint8)

def saveNIIrgb(V, path):
    shape_3d = V.shape[0:3]
    rgb_dtype = np.dtype([('R', 'u1'), ('G', 'u1'), ('B', 'u1')])
    V = V.copy().view(rgb_dtype).reshape(shape_3d)
    img = nib.Nifti1Image(V, np.eye(4))

    nib.save(img, str(path))

def saveNII(V, path):
    img = nib.Nifti1Image(V, np.eye(4))
    nib.save(img, str(path))



data_str='_rgb.nii.gz'
label_str='_l.nii.gz'


root_dir = '/home/gerlstefan/data/fullDataset/slice_shuffle_backup'

assert os.path.exists(root_dir) and os.path.isdir(root_dir), \
root_dir + ' not a valid directory'

assert isinstance(data_str, str) and isinstance(label_str, str), \
'data_str or label_str not valid.'

# get all files in train_dir
files = os.listdir(path = root_dir)
# extract the data files
all_data = [el for el in files if el[-len(data_str):] == data_str]

assert len(all_data) == \
    len([el for el in files if el[-len(label_str):] == \
    label_str]), \
    'Amount of data and label files not equal.'

if len(all_data) == 0:
    print('No data in ' + root_dir + ' found.')

print('Found', len(all_data), 'volumes')

# now load all .nii files and stack together.
for ctr, file in enumerate(all_data):
    if ctr == 1:
        data_stack = data
        label_stack = label

    data = readData(os.path.join(root_dir, file))
    print(data.shape)

    label = readLabel(os.path.join(root_dir, file.replace(data_str, label_str)))
    assert np.amax(label) == 1
    print(label.shape)

    if ctr >= 1:
        data_stack = np.concatenate((data_stack, data), axis=1)
        label_stack = np.concatenate((label_stack, label),axis=1)

print('After concatenation')
print(data_stack.shape, '\n', label_stack.shape)

# swap axes in order to shuffle slices
data_stack = np.swapaxes(data_stack, 0, 1)
label_stack = np.swapaxes(label_stack, 0, 1)
print('After swapping axes 0 and 1')
print(data_stack.shape, '\n', label_stack.shape)

# shuffle .nii files along first dimension

rnd_seed = 568;
np.random.seed(rnd_seed)
np.random.shuffle(data_stack)

np.random.seed(rnd_seed)
np.random.shuffle(label_stack)

# swap axes back
data_stack = np.swapaxes(data_stack, 0, 1)
label_stack = np.swapaxes(label_stack, 0, 1)


# split into one training and one validation set
# 20 training vol + 5 validation vol = 25
# 20/25 = 0.8
split_idx = int(0.8*data_stack.shape[1])


train_data_stack = data_stack[:,:split_idx,:,:]
train_label_stack = label_stack[:,:split_idx,:]

val_data_stack = data_stack[:,split_idx:,:,:]
val_label_stack = label_stack[:,split_idx:,:]

print('After split:')
print('Train:', train_data_stack.shape, '\n', train_label_stack.shape)
print('Val:', val_data_stack.shape, '\n', val_label_stack.shape)

# save

dest = '/home/gerlstefan/data/fullDataset/slice_shuffle'

saveNIIrgb(train_data_stack, os.path.join(dest, 'train', 'train_rgb.nii.gz'))
saveNII(train_label_stack, os.path.join(dest, 'train', 'train_l.nii.gz'))

saveNIIrgb(val_data_stack, os.path.join(dest, 'val', 'val_rgb.nii.gz'))
saveNII(val_label_stack, os.path.join(dest, 'val', 'val_l.nii.gz'))


