#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 29 15:49:45 2019

@author: stefan
"""
import torch
import copy
import math

from skimage import morphology
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

    S = meta['label_skeleton']
    S = S.to('cuda', dtype=torch.bool)
    
    #debug
    # pred = copy.deepcopy(target)
    # end debug!!!!!
    # print('minmax pred', torch.min(pred).item(), torch.max(pred).item())
    
    pred = pred.detach()
    pred = pred.to(torch.bool)
    target = target.to(torch.bool)

    # calculate centerline score
    # number of pixels of sceleton inside pred / number of pixels in sceleton
    try:
        nom =  torch.sum(S & pred, dtype=torch.float32)
        denom = torch.sum(S, dtype=torch.float32)

        # print('cl_score', nom, '/', denom)
        cl_score = nom / denom
    except ZeroDivisionError:
        # if the sceleton is zero, the score is automatically 1??
        print('ZeroDivisionError in cl_score')
        cl_score = torch.tensor([1], dtype=torch.float32)

    cl_score = cl_score.to(device='cpu')

    # dilate target/label massive
    # to generate hull
    ball_r = 5 
    element = morphology.ball(ball_r) # good value seems in between 3 and 5
    element = torch.from_numpy(element).to('cuda', dtype=torch.float32)
    element = torch.unsqueeze(torch.unsqueeze(element, 0), 0)

    # dilation: use torch conv3d
    H = torch.nn.functional.conv3d(target.to(dtype=torch.float32), element, padding=ball_r)
    H = H >= 1
    # 1 - number of pixels of prediction outside hull / number of pixels of prediction inside hull ? 
    # or just total number of pixels of prediction
    nom = torch.sum(~H & pred, dtype=torch.float32)
    denom = torch.sum(pred, dtype=torch.float32)
    out_score = 1 - nom/denom
    out_score = out_score.to('cpu')
    # print('out_score', nom, '/', denom)
    # print('cl_score', cl_score.item(), 'out_score', out_score.item())
    
    # multiply with batch size!
    batch_size = pred.shape[0]
    # print('Batch size:', batch_size)
    cl_score = batch_size * cl_score.item()
    out_score = batch_size * out_score.item()

    smooth = 1.

    intersection = torch.sum(pred & target, dtype=torch.float32)
    dice = (2 * intersection + smooth) / (pred.float().sum() + target.float().sum() + smooth)
    # print('dice', dice)
    dice = dice.to('cpu').item()


    return {'cl_score': cl_score,
            'out_score': out_score,
            'dice': dice }
