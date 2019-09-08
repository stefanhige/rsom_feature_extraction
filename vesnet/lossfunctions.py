#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 29 15:49:45 2019

@author: stefan
"""
import torch
import copy
import math

def BCEWithLogitsLoss(pred, target, weight=None):

    fn = torch.nn.BCEWithLogitsLoss(weight=None,
                                    reduction='mean',
                                    pos_weight=None)

    loss = fn(pred, target)
    
    return loss



def calc_metrics(pred, target, meta):
    """
    calculate metrics e.g. dice, centerline score
    """

    # what happens if batchsize is not 1?
    # better calculate skeleton in dataloader
    # S = meta['label_skeleton']
    S = meta 

    print(S.dtype)
    print(S.shape)

    S = S.to(torch.uint8)
    # missing, convert from probability to bool
    pred = pred.detach()
    pred = pred.to(torch.uint8)
    target = target.to(torch.uint8)

    # byte tensor supports logical and
    # need to shrink shape of S and label make fit to pred
    # only looking ad valid predictions
    # not yet working
    # need to adjust size:

    print(int(math.floor(d[2]/2)), -int(math.ceil(d[2]/2)))
    target = target[:,:,int(math.floor(shp[2]/2)):-int(math.ceil(shp[2]/2)),:,:]
    print('Pred shape:', pred.shape)
    print('Target shape:', target.shape)
    # calculate centerline score
    # number of pixels of sceleton inside pred / number of pixels in sceleton
    cl_score = torch.sum(S & pred) / torch.sum(S)
    cl_score = cl_score.to(device='cpu')

    # dilate label massive
    # to generate hull

    element = morphology.ball(5) # good value seems in between 3 and 5
    element = torch.from_numpy(element).to(dtype=torch.uint8)

    # use torch conv3d
    label = label.to(dtype=torch.uint8)
    H = torch.nn.conv3d(label, element, padding=4)

    H = ndimage.morphology.binary_dilation(label, iterations=1, structure=element)

    # 1 - number of pixels of prediction outside hull / number of pixels of prediction inside hull ? 
    # or just total number of pixels of prediction
    out_score = 1 - np.count_nonzero(np.logical_and(np.logical_not(H), pred)) / np.count_nonzero(pred)


# pred = torch.zeros((3,1,10,10,10))
# pred[:, :, 5, 5, 5] = 1

# target = torch.zeros((3,1,7,7,7))
# target[:, :, 5, 5, 5] = 1


# meta = copy.deepcopy(target)

# metrics = calc_metrics(pred, target, meta)



