import torch
import warnings
# from torch import nn
import copy 
import numpy as np
import math
def cross_entropy_2d(pred, target, weight):
    '''
    doc string
    '''
    fn = torch.nn.CrossEntropyLoss()
    
    ps = pred.shape
    ts = target.shape

    assert ps[0] == ts[0], 'Batch sizes incorrect.'

    # in case of incorrect padding, d1 and d2 
    # of pred and target may be different
    if (ps[2] != ts[1]) or ps[3] != ts[2]:
        warnings.warn("Shapes of Prediction and Target do not match.",
                'Pred', ps, 'Target', ts)
        p_d1 = ps[2]
        p_d2 = ps[3]
        t_d1 = ps[1]
        t_d2 = ps[2]

        # half of prediction
        # TODO: implement symmetric cutting
        target = target[:,:p_d1,:p_d2]

    # typecast
    pred = pred.float()
    target = target.long()

    # pred shape is
    # [N x C x d1 x d2]
    # N: batch size
    # C: number of Classes (here: 2)
    # d1 d2: spatial dimensions
    # dtype: float

    # target shape is
    # [N x d1 x d2]
    # with N[,,,] = 0...C-1 (here: either 0 or 1)
    # entries are class labels
    # dtype: long
    loss = fn(pred, target)

    return loss

def custom_loss_1(pred, target, spatial_weight, class_weight=None):
    '''
    doc string
    '''
    fn = torch.nn.CrossEntropyLoss(weight=class_weight, reduction='none')
    
    pred = pred.float()
    target = target.long()
    unred_loss = fn(pred, target)

    loss = spatial_weight.float() * unred_loss
    loss = torch.sum(loss)
    
    return loss 
 
def custom_loss_1_smooth(pred, target, spatial_weight, class_weight=None, smoothness_weight=100):

    fn = torch.nn.CrossEntropyLoss(weight=class_weight, reduction='none')
    
    pred = pred.float()
    target = target.long()
    unred_loss = fn(pred, target)

    loss = spatial_weight.float() * unred_loss
    loss = torch.sum(loss)
    
    # add smoothness loss
    more =  smoothness_weight*smoothness_loss(pred)
    loss += more
    
    return loss 
 
def scalingfn(l):
    '''
    l is length
    '''
    # linear, starting at 1
    y = torch.arange(l) + 1
    return y

def smoothness_loss(pred):
    '''
    smoothness loss x-y plane, ie. perfect label
    separation in z-direction will cause zero loss
    
    first try only calculating the loss on label "1"
    as this is a 2-label problem only anyways
    '''
    pred_shape = pred.shape 
   
    # this gives nonzero entries for label "1"
    label = (pred[:,1,:,:] - pred[:,0,:,:]).float()
    label = torch.nn.functional.relu(label)

    label = label.view(-1)

    window = 5

    # add 2 extra dimensions
    # conv1d needs input of shape
    # [minibatch x in_channels x iW]
    label = torch.unsqueeze(label, 0)
    label = torch.unsqueeze(label, 0)

    # weights of the convolutions are simply 1, and divided by the window size
    weight = torch.ones(1, 1, window).float().to('cuda') / window

    label_conv = torch.nn.functional.conv1d(input=label, 
            weight=weight,
            padding=int(math.floor(window/2)))
    
    label_conv = torch.squeeze(label_conv)
    label = torch.squeeze(label)
    
    # for perfectly smooth label, this value is zero
    # e.g. if label_conv[i] = label[i], -> 1/1 - 1 = 0
    label_smoothness =torch.abs((label_conv+1) / (label+1)-1)
    
    # edge correction, steps at the boundaries do not count as unsmooth,
    # therefore corresponding entries of label_smoothness are zeroed out
    edge_corr = torch.zeros((pred_shape[3])).to('cuda')
    edge_corr[int(math.floor(window/2)):-int(math.floor(window/2))] = 1
    edge_corr = edge_corr.repeat(pred_shape[0]*pred_shape[2])

    label_smoothness *= edge_corr
    
    # target shape
    # [minibatch x Z x X]
    
    # return some loss measure, as the sum of all smoothness losses
    return torch.sum(label_smoothness)    
    
