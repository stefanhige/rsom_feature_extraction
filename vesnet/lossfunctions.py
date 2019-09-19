#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 29 15:49:45 2019

@author: stefan
"""
import torch
import copy
import math
import numpy as np
from scipy.optimize import minimize_scalar

from skimage import morphology
def BCEWithLogitsLoss(pred, target, weight=None):

    fn = torch.nn.BCEWithLogitsLoss(weight=None,
                                    reduction='mean',
                                    pos_weight=torch.Tensor([weight]).cuda())
    loss = fn(pred, target)
    
    return loss



def dice_loss(pred, target, eps=1e-7):
    """Computes the Sørensen–Dice loss.
    from:
    https://github.com/kevinzakka/pytorch-goodies

    Args:
        true: a tensor of shape [B, 1, H, W].
        logits: a tensor of shape [B, C, H, W]. Corresponds to
            the raw output or logits of the model.
        eps: added to the denominator for numerical stability.
    Returns:
        dice_loss: 1-dice 
    """
    # parse
    target = target.long()

    num_classes = target.shape[1]
    if num_classes == 1:
        #not clear whats happening here
        # adds dimension at the end, one being the negation of the other
        # order [Foreground, Background)
        mask = (torch.eye(2)[torch.tensor([1, 0]).long()])[target.squeeze(1)]
        # move to axis=1
        mask = mask.permute(0, 4, 1, 2, 3).float()
        
        pos_prob = torch.sigmoid(pred)
        neg_prob = 1 - pos_prob
        prob = torch.cat([pos_prob, neg_prob], dim=1)
    else:
        raise NotImplementedError
    
    mask = mask.type_as(pred)
    dims = (0,) + tuple(range(2, target.ndimension()))
    intersection = torch.sum(prob * mask, dims)
    cardinality = torch.sum(prob + mask, dims)
    dice_loss = (2. * intersection + eps / (cardinality + eps)).mean()
    return (1 - dice_loss)

if __name__ == '__main__':

    import os
    os.environ["CUDA_VISIBLE_DEVICES"]='7'
    torch.random.manual_seed(5)
    P = torch.rand((1, 1, 3, 3, 3), requires_grad = True)
    logits = torch.log(P/(1-P))

    label = torch.rand((1, 1, 3, 3, 3))
    label = (label >= 0.5).float()

    dice = dice_loss(logits, label)


def find_cutoff(pred, label):
    '''
    find ideal binary cutoff to maximize dice
    '''
    print('function find_cutoff')
    print(pred.shape)
    print(label.shape)

    # minimize 1- dice
    def _fun(x):
        print(pred.shape)
        pred_bool = pred >= x
        dice = _dice(pred_bool, label)
        print('x:', x, 'dice:', dice)
        return 1 - dice

    res = minimize_scalar(_fun, bounds=(0, 1), method='bounded')
    return res.x, 1-_fun(res.x) 

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

    i = np.logical_and(x, y)

    if x.sum() + y.sum() == 0:
        print('no True elements in this patch!')
        return 1.

    return (2. * i.sum()) / (x.sum() + y.sum())
    
def calc_metrics(pred, target, skel):
    """
    calculate metrics e.g. dice, centerline score
    """

    S = skel
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
    
    nom =  torch.sum(S & pred, dtype=torch.float32)
    denom = torch.sum(S, dtype=torch.float32)
    if denom == 0:
        print('Skeleton empty, cl_score=nan')
        cl_score = float('nan')
    else:
        cl_score = nom / denom
        cl_score = cl_score.to(device='cpu').item()

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
    # this was a bad idea! wrong!
    batch_size = pred.shape[0]
    # print('Batch size:', batch_size)
    # cl_score = batch_size * cl_score.item()
    # out_score = batch_size * out_score.item()

    tensor_sum = pred.float().sum() + target.float().sum()
    if tensor_sum == 0:
        print('Warning, tensor_sum is zero, dice will be nan')
        dice = float('nan')
    else:
        intersection = torch.sum(pred & target, dtype=torch.float32)
        dice = (2 * intersection) / tensor_sum
        # print('dice', dice)
        dice = dice.to('cpu').item()

    return cl_score, out_score.item(), dice
