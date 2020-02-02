#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 25 17:21:21 2020

@author: stefan
"""
import nibabel as nib
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch
import os
# from mpl_toolkits.mplot3d import Axes3D

import matplotlib.cm as cm
from scipy.ndimage.filters import gaussian_filter1d 

from scipy.ndimage.filters import uniform_filter1d


def load_seg(path):
    img = nib.load(path)
    return img.get_data()
    
def load_rgb(path):
    img = nib.load(path)
    data = img.get_data()
    data = np.stack([data['R'], data['G'], data['B']], axis=-1)
    return data

def save_nii(V, path, fstr=None):
    
    if fstr is not None:
        if not '.nii.gz' in fstr:
            fstr += '.nii.gz'
        path = os.path.join(path, fstr)
    
    #V = V.astype(np.uint8)
    img = nib.Nifti1Image(V, np.eye(4))
    nib.save(img, path)
    
def extract_surf(P):    
    Pproj = np.sum(P, axis=2)
    
    Pproj_ = gaussian_filter1d(Pproj, sigma=7, axis=1)
    
    Maxima = np.argmax(Pproj_, axis=0)
    surf1 = np.zeros((P.shape[1], P.shape[2]))
    surf2 = np.zeros_like(surf1)
    
    zerorough=0
    for y in np.arange(P.shape[1]):
        for x in np.arange(P.shape[2]):
            nz = np.nonzero(P[:, y, x])
            nz = nz[0]
            #print(nz)
            # only valid if we captured the whole thing
            if len(nz) == 0:    
                surf1[y,x] = Maxima[y] #+ int(np.random.normal(scale=3))
                surf2[y,x] = Maxima[y] #+ int(np.random.normal(scale=3))
                zerorough += 1
                # surf1[y,x] = np.nan
                # surf2[y,x] = np.nan
                # print('no epidermis, value at', x, y, 'set to', Maxima[y])
            else:
                if nz[0] == 0:
                    print('inside the layer!',x,y)
                    surf1[y,x] = Maxima[y] #+ int(np.random.normal(scale=3))
                    surf2[y,x] = Maxima[y] #+ int(np.random.normal(scale=3))
                    pass
                else:
                    # TODO check if holes
                    cutoff = 15
                    if np.any(np.diff(nz)>cutoff):
                        nzidx = np.nonzero((np.diff(nz)>cutoff -1).astype(np.bool))
                        nzidx = nzidx[0]
                        if len(nzidx) > 1:
                            #TODO these cases are ignored 
                            start = nz[0]
                            end = nz[-1]
                            # print(nz)
                            # print(nzidx)
                        elif nzidx < 0.2*len(nz):
                            # print('take last part')
                            # print(nz)
                            # print(nzidx)
                            start = nz[nzidx+1]
                            end = nz[-1]
                            # print(start, end)
                        elif nzidx > 0.8*len(nz):
                            # print('take first part')
                            # print(nz)
                            # print(nzidx)
                            start = nz[0]
                            end = nz[nzidx]
                            # print(start, end)
                        else:
                            start = nz[0]
                            end = nz[-1]
                    else:
                        start = nz[0]
                        end = nz[-1]
                    surf1[y,x] = start
                    surf2[y,x] = end
                    # surf1[y,x] = nz[0]
                    # surf2[y,x] = nz[-1]
    return surf1, surf2, zerorough                 
    
def rms_roughness(surf, window=5):
    mv_avg_surf = uniform_filter1d(surf, window, axis=1)
    

    # res_surf = np.sqrt(np.mean(np.power(surf1 - mv_avg_surf, 2),axis=1))
    res_surf = np.mean(np.abs(surf-mv_avg_surf),axis=1)
    # print(res_surf.shape)
    return res_surf
 
def plot_surface(S, dest, title=''):
    '''
    plot the surfaceData used for the normalization
    mainly for debugging purposes
    '''
    
    plt.ioff()
    plt.figure()
    
    Surf = S-np.amin(S)
    
    plt.imshow(Surf, cmap=cm.jet)
    plt.colorbar()
    plt.title(title) 
    plt.savefig(dest)
        
    plt.close()

window=5
print('window',window)

dirs = ['/home/gerlstefan/data/layerunet/miccai/200201-04-BCE',
        '/home/gerlstefan/data/layerunet/miccai/200202-02-BCE_S_1',
        '/home/gerlstefan/data/layerunet/miccai/200202-03-BCE_S_10',
        '/home/gerlstefan/data/layerunet/miccai/200202-04-BCE_S_100']
# dirs = ['/home/gerlstefan/data/layerunet/miccai/200201-04-BCE',
#         '/home/gerlstefan/data/layerunet/miccai/200201-05-BCE_W',
#         '/home/gerlstefan/data/layerunet/miccai/200201-06-BCE_W_S_1',
#         '/home/gerlstefan/data/layerunet/miccai/200201-07-BCE_W_S_10',
#         '/home/gerlstefan/data/layerunet/miccai/200202-00-BCE_W_S_100',
#         '/home/gerlstefan/data/layerunet/miccai/200202-01-BCE_W_S_1000'
#         ]
for loc in dirs:
    files = os.listdir(loc)
    files = [el for el in files if "_pred.nii.gz" in el]
    P = []
    for f in files:
        P.append(load_seg(os.path.join(loc,f)))
    P = np.concatenate(P, axis=1)
    surf1, surf2, n_holes = extract_surf(P)
    # save surfaces as images
    plot_surface(surf1, os.path.join(loc,'surf1.png')) 
    plot_surface(surf2, os.path.join(loc,'surf2.png')) 
    
    surf1_rms = rms_roughness(surf1,window=window)
    surf2_rms = rms_roughness(surf2,window=window)
    rms = np.concatenate((surf1_rms, surf2_rms), axis=0)
    print(os.path.basename(loc),'holes', n_holes, 'roughness', np.mean(rms))
end
file1 = os.path.join(dirr1,filename)
file1p5 = os.path.join(dirr1p5,filename)
file2 = os.path.join(dirr2,filename)
file3 = os.path.join(dirr3,filename)
file4 = os.path.join(dirr4,filename)

file_gt = os.path.join('/home/stefan/fbserver_ssh/data/layerunet/fullDataset/labeled/val/',
        filename.replace('pred','l'))
file1_fft, _ = calc_surf_fft(file1)
file1p5_fft, _ = calc_surf_fft(file1p5)
file2_fft, _ = calc_surf_fft(file2)
file3_fft, a = calc_surf_fft(file3) 
file4_fft, b = calc_surf_fft(file4) 


#idx =10
#
#fig, ax = plt.subplots()
##ax.plot(np.abs(a[idx,:]),label='1000')
##ax.plot(np.abs(b[idx,:]))
#ax.plot(abs(file3_fft),label='1000mean')
#ax.plot(abs(file4_fft))
#ax.set_yscale('log')
#ax.legend()
#
#end

if 1:
    sigma=5
    
    file1_fft_smooth = gaussian_filter1d(np.abs(file1_fft),sigma)
    file1p5_fft_smooth = gaussian_filter1d(np.abs(file1p5_fft),sigma)
    file2_fft_smooth = gaussian_filter1d(np.abs(file2_fft),sigma)
    file3_fft_smooth = gaussian_filter1d(np.abs(file3_fft),sigma)
    file4_fft_smooth = gaussian_filter1d(np.abs(file4_fft),sigma)
else:
    sigma=0.1
    file1_fft_smooth = gaussian_filter1d(np.abs(file1_fft),sigma)
    file2_fft_smooth = gaussian_filter1d(np.abs(file2_fft),sigma)
    file3_fft_smooth = gaussian_filter1d(np.abs(file3_fft),sigma)
    file4_fft_smooth = gaussian_filter1d(np.abs(file4_fft),sigma)
    

file1_fft_smooth *= 1/np.max(file1_fft_smooth) 
file1p5_fft_smooth *= 1/np.max(file1p5_fft_smooth) 
file2_fft_smooth *= 1/np.max(file2_fft_smooth) 
file3_fft_smooth *= 1/np.max(file3_fft_smooth) 
file4_fft_smooth *= 1/np.max(file4_fft_smooth) 


av1 = moving_avg(file1)

av1p5 = moving_avg(file1p5)

av2 = moving_avg(file2)

av3 = moving_avg(file3)

av4 = moving_avg(file4)
#f=np.linspace(-1,1,num=len(file1_fft))
f = np.arange(len(file1_fft))
            
fig, ax = plt.subplots()
ax.plot(f,file1_fft_smooth, label='s=0')
ax.plot(f,file1p5_fft_smooth, label='s=1')
ax.plot(f,file2_fft_smooth, label='s=100')
ax.plot(f,file3_fft_smooth,label='s=1000')
ax.plot(f,file4_fft_smooth,label='s=10000')
ax.set_yscale('log')
ax.legend()
ax.set(xlabel='f', ylabel='|Intensity| a.u.')
plt.show()
#ax.set_ylim(1e-1, 1e2)


def _dice(x, y):
    '''
    do the test in numpy
    '''
    if isinstance(x, torch.Tensor):
        x = x.cpu().numpy()
    if isinstance(y, torch.Tensor):
        y = y.cpu().numpy()

    x = x.astype(np.bool)
    y = y.astype(np.bool)

    i = np.logical_and(x,y)

    if x.sum() + y.sum() == 0:
        return 1.

    return (2. * i.sum()) / (x.sum() + y.sum())




seg1 = load_seg(file1)
seg1p5 = load_seg(file1p5)
seg2 = load_seg(file2)
seg3 = load_seg(file3)
seg4 = load_seg(file4)
gt = load_seg(file_gt)

dice1 = _dice(seg1,gt)
dice1p5 = _dice(seg1p5,gt)
dice2 = _dice(seg2,gt)
dice3 = _dice(seg3,gt)
dice4 = _dice(seg4,gt)

#approximate indices

idx = int(len(file3_fft)*0.4)

pow1 = np.trapz(np.power(file1_fft_smooth[:idx],2))
pow1p5 = np.trapz(np.power(file1p5_fft_smooth[:idx],2))
pow2 = np.trapz(np.power(file2_fft_smooth[:idx],2))
pow3 = np.trapz(np.power(file3_fft_smooth[:idx],2))
pow4 = np.trapz(np.power(file4_fft_smooth[:idx],2))


fig, ax1 = plt.subplots()

color = 'tab:red'
ax1.set_xlabel('s')
ax1.set_ylabel('Dice', color=color)
ax1.plot([0, 1, 100, 1000], [dice1, dice1p5, dice2, dice3], color=color, marker=10)
ax1.tick_params(axis='y', labelcolor=color)
ax1.set_ylim(0.5, 1)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color = 'tab:blue'
ax2.set_ylabel('RMS roughness', color=color)  # we already handled the x-label with ax1
poww = np.array([pow1, pow1p5, pow2, pow3])
avv = np.array([av1, av1p5, av2, av3])

ax2.plot([0, 1, 100, 1000],avv / avv.max(), color=color, marker=11)
# ax2.plot([0, 1, 100, 1000, 10000],poww / poww.max(), color=color, marker=11)
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.show()



        
    


        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        


