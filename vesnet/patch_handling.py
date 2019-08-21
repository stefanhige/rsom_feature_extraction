#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 21 14:09:13 2019

@author: stefan
"""

import numpy as np

def get_patches(volume, divs, offset):
    if isinstance(divs, int):
        divs = (divs,)
    if isinstance(offset, int):
        offset = (offset,)
    
    assert len(volume.shape) == len(divs)
    assert len(volume.shape) == len(offset)

    patches = []
    for idx in np.arange(np.prod(divs)):
        patches.append(get_patch_ndim(volume, idx, divs, offset))
    
    return np.array(patches)


def get_patch_ndim(volume, index, divs=(2,2,2), offset=(6,6,6)):
    '''
    Args:
        - volume3d (np.array)       :   The volume to cut
        - index (int)               :   in range 0 to sum(divs)-1
        
        - divs (tuple, optional)    :   Amount to divide each side
        - offset (tuple, optional)  :   Offset for each div
    '''
    if isinstance(divs, int):
        divs = (divs,)
    if isinstance(offset, int):
        offset = (offset,)
    
    assert len(volume.shape) == len(divs)
    assert len(volume.shape) == len(offset)
    
    shape = volume.shape
    widths = [int(s/d) for s, d in zip(shape, divs)]
    patch_shape = [w+o*2 for w, o in zip(widths, offset)]
    index_ = np.unravel_index(index, divs)
    #print(index_)
    # coords
    c = [s*d for s, d in zip(index_, widths)]
    #print(c) 
    patch = np.zeros(patch_shape, dtype=volume.dtype)
    s_ = []
    e_ = []
    slice_idx = []
    slice_idx_patch = []
    for dim in np.arange(len(c)):
        s_ = c[dim] - offset[dim] if c[dim] - offset[dim] >= 0 else 0
        e_ = c[dim] + widths[dim] + offset[dim] if \
            c[dim] + widths[dim] + offset[dim] <= shape[dim] else shape[dim]
        slice_idx.append(slice(s_, e_))
        
        ps_ = offset[dim] - (c[dim] - s_)
        pe_ = ps_ + (e_ - s_)
        slice_idx_patch.append(slice(ps_, pe_))

    # print(slice_idx) 
    slice_idx = tuple(slice_idx)
    slice_idx_patch = tuple(slice_idx_patch)
    
    vp = volume[slice_idx]
    patch[slice_idx_patch] = vp
    return patch


# not finished.
def get_volume(patches, divs = (2,2,3), offset=(6,6,6)):
    """
    """
    if isinstance(divs, int):
        divs = (divs,)
    if isinstance(offset, int):
        offset = (offset,)

    new_shape = [(ps -of*2)*int(d) for ps, of, d in zip(patches.shape[1:], offset, divs)]
    volume = np.zeros(new_shape, dtype=patches.dtype)
    shape = volume.shape
    widths = [int(s/d) for s, d in zip(shape, divs)]
    for index in np.arange(np.prod(divs)):
        index_ = np.unravel_index(index, divs)
        slice_idx = []
        slice_idx_offs = []
        for dim in np.arange(len(index_)):
            # print(index_)
            s_ = (index_[dim] * widths[dim])
            e_ = ((index_[dim] + 1) * widths[dim])
            slice_idx.append(slice(s_, e_))

            s__ = offset[dim]
            e__ = offset[dim] + widths[dim]
            slice_idx_offs.append(slice(s__, e__))
            
        patch = patches[index,...]
        volume[tuple(slice_idx)] = patch[tuple(slice_idx_offs)]
    return volume


def get_single_patch(volume, index, divs=(2,2,2), offset=(6,6,6)):
    '''
    Generate minibatches, by Giles Tetteh
    Args:
        - volume3d (np.array)       :   The volume to cut
        - index (int)               :   in range 0 to sum(divs)-1
        
        - divs (tuple, optional)    :   Amount to divide each side
        - offset (tuple, optional)  :   Offset for each div
    '''
    assert len(volume.shape) == len(divs)
    assert len(volume.shape) == len(offset)
    
    shape = volume.shape
    widths = [int(s/d) for s, d in zip(shape, divs)]
    patch_shape = [w+o*2 for w, o in zip(widths, offset)]
    index_ = np.unravel_index(index, divs)
    print(index_)
    
    # coords
    x, y, z = [s*d for s, d in zip(index_, widths)]
    
    patch = np.zeros(patch_shape, dtype=volume.dtype)
    x_s = x - offset[0] if x - offset[0] >= 0 else 0
    x_e = x + widths[0] + offset[0] if x + \
                        widths[0] + offset[0] <= shape[0] else shape[0]
        
    y_s = y - offset[1] if y - offset[1] >= 0 else 0
    y_e = y + widths[1] + offset[1] if y + widths[1] + offset[1] <= shape[1] else shape[1]
    z_s = z - offset[2] if z - offset[2] >= 0 else 0
    z_e = z + widths[2] + offset[2] if z + widths[2] + offset[2] <= shape[2] else shape[2]
    vp = volume[x_s:x_e,y_s:y_e,z_s:z_e]
    print(vp.shape)
    px_s = offset[0] - (x - x_s)
    px_e = px_s + (x_e - x_s)
    py_s = offset[1] - (y - y_s)
    py_e = py_s + (y_e - y_s)
    pz_s = offset[2] - (z - z_s)
    pz_e = pz_s + (z_e - z_s)
    patch[px_s:px_e, py_s:py_e, pz_s:pz_e] = vp
    
    return patch
    
    
    
    


def get_patch_data3d(volume3d, divs=(2,2,2), offset=(6,6,6)):
    """Generate minibatches, by Giles Tetteh
    Args:
        - volume3d (np.array)       :   The volume to cut
        - divs (tuple, optional)    :   Amount to divide each side
        - offset (tuple, optional)  :   Offset for each div
    """
    patches = []
    shape = volume3d.shape
    widths = [int(s/d) for s, d in zip(shape, divs)]
    patch_shape = [w+o*2 for w, o in zip(widths, offset)]
    #print("V3dshape {}".format(volume3d.shape))

    for x in np.arange(0, shape[0], widths[0]):
        for y in np.arange(0, shape[1], widths[1]):
            for z in np.arange(0, shape[2], widths[2]):
                print(x,y,z)
                patch = np.zeros(patch_shape, dtype=volume3d.dtype)
                x_s = x - offset[0] if x - offset[0] >= 0 else 0
                x_e = x + widths[0] + offset[0] if x + \
                        widths[0] + offset[0] <= shape[0] else shape[0]
                y_s = y - offset[1] if y - offset[1] >= 0 else 0
                y_e = y + widths[1] + offset[1] if y + \
                        widths[1] + offset[1] <= shape[1] else shape[1]
                z_s = z - offset[2] if z - offset[2] >= 0 else 0
                z_e = z + widths[2] + offset[2] if z + \
                        widths[2] + offset[2] <= shape[2] else shape[2]

                vp = volume3d[x_s:x_e,y_s:y_e,z_s:z_e]
                px_s = offset[0] - (x - x_s)
                px_e = px_s + (x_e - x_s)
                py_s = offset[1] - (y - y_s)
                py_e = py_s + (y_e - y_s)
                pz_s = offset[2] - (z - z_s)
                pz_e = pz_s + (z_e - z_s)
                patch[px_s:px_e, py_s:py_e, pz_s:pz_e] = vp
                patches.append(patch)
    
    #return patches

    return np.array(patches, dtype = volume3d.dtype)


def get_volume_from_patches3d(patches4d, divs = (2,2,3), offset=(6,6,6)):
    """Reconstruct the minibatches, by Giles Tetteh
    Keep offset of (0,0,0) for fully padded volumes
    """
    new_shape = [(ps -of*2)*int(d) for ps, of, d in zip(patches4d.shape[-3:], offset, divs)]
    volume3d = np.zeros(new_shape, dtype=patches4d.dtype)
    shape = volume3d.shape
    widths = [int(s/d) for s, d in zip(shape, divs)]
    index = 0
    for x in np.arange(0, shape[0], widths[0]):
        for y in np.arange(0, shape[1], widths[1]):
            for z in np.arange(0, shape[2], widths[2]):
                patch = patches4d[index]
                index = index + 1
                volume3d[x:x+widths[0],y:y+widths[1],z:z+widths[2]] = \
                        patch[offset[0]:offset[0] + widths[0], offset[1]:offset[1]+widths[1], offset[2]:offset[2]+widths[2]]
    return volume3d



# A = np.random.random_sample((100, 100, 100))
# patches = []
# for idx in np.arange(8):
#     print(idx)
#     A_1patch = get_single_patch(A, idx)
#     A_1patch_new = get_patch_ndim(A, idx)
#     if not np.all(A_1patch == A_1patch_new):
#         print('error')
        
#     patches.append(A_1patch_new)
    
# patches = np.array(patches)

# patches_giles = get_patch_data3d(A)

# if not np.all(patches == patches_giles):
#         print('error')

#A_patched = get_patch_data3d(A)

#A_rec = get_volume_from_patches3d(A_patched)
        


