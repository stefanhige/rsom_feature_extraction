#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 25 18:39:48 2020

@author: stefan
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter1d

a = np.load('data1.npy')

b = np.load('data2.npy')




c = b + 10*np.sin(np.arange(len(b)))

dft = np.fft.fft

af = dft(a)
bf = dft(b)
cf = dft(c)

#signals
fig, ax = plt.subplots()
ax.plot(a)
ax.plot(b)


f = np.fft.fftfreq(len(a), d=1)
# ffts
fig, ax = plt.subplots()
ax.plot(f,np.abs(af))
ax.plot(f,np.abs(bf))

#ax.plot(f,np.abs(cf))

ax.set_ylim(0, 300)


# ffts
fig, ax = plt.subplots()
sigma=3
ax.plot(f,gaussian_filter1d(np.abs(af),sigma))
ax.plot(f,gaussian_filter1d(np.abs(bf),sigma))
ax.set_ylim(0, 300)

ax.set(xlabel='f', ylabel='|Intensity| a.u.')
