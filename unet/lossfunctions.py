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
    # fn = torch.nn.CrossEntropyLoss(weight=torch.tensor([0.1, 0.9]).to('cuda'))
    fn = torch.nn.CrossEntropyLoss()
    # fn = torch.nn.CrossEntropyLoss(reduction='none')
    
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

    # print('LOSS SHAPE:', loss.shape)
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
    # print('L COMP', loss)
    # print(loss.requires_grad)
    # more =  100*smoothness_loss(pred)
    # print('S COMP', more)
    # loss += more
    # print(more.requires_grad)
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
    # print('prediction shape', pred.shape)
    label = (pred[:,1,:,:] - pred[:,0,:,:]).float()
    
    label = torch.nn.functional.relu(label)
    #add reLU?
    
    #look at values?

    
    label = label.view(-1)
    # print(label.shape)
    # print(label)

    window = 5

    # add 2 extra dimensions
    # conv1d needs input of shape
    # [minibatch x in_channels x iW]
    label = torch.unsqueeze(label, 0)
    label = torch.unsqueeze(label, 0)
    # print('label after unsqueeze')
    # print(label.shape)

    # weight 
    weight = torch.ones(1, 1, window).float().to('cuda') / window
    # print('conv weight:', weight)

    label_conv = torch.nn.functional.conv1d(input=label, 
            weight=weight,
            padding=int(math.floor(window/2)))
    
    label_conv = torch.squeeze(label_conv)
    label = torch.squeeze(label)
    # print('shapes after conv:', label_conv.shape, label.shape)
    label_smoothness =torch.abs((label_conv+1) / (label+1)-1)
    # print(label_smoothness)
    # print('sum label_smoothness', torch.sum(label_smoothness))

    edge_corr = torch.zeros((pred_shape[3])).to('cuda')
    edge_corr[int(math.floor(window/2)):-int(math.floor(window/2))] = 1
    # print(edge_corr)
    edge_corr = edge_corr.repeat(pred_shape[0]*pred_shape[2])
    # print(edge_corr)

    label_smoothness *= edge_corr
    # print('edge corrected label smoothness')
    # print(label_smoothness)
    # print(torch.sum(label_smoothness))
    # target shape
    # [minibatch x Z x X]
    return torch.sum(label_smoothness)    
    
    # torch.nn.functional.conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1, padding_mode='zeros')
    # input – minibatch,in_channels,iH,iW
    # weight filters of shape out_channels,in_channels/groups,kH,kW

    # so Z needs to be minibatch, in_channels = 1, iH = minibatch, iW = X
    # target needs to be reshaped to
    # [Z, 1, minibatch, X]
    # weight needs to be defined as
    # [1, 1, minibatch, minibatch]
    # options: padding=0, groups=1, dilation=1
    # maybe add ReLu or abs() function before (need to check pred values!)


    # then to calculate loss
    # for each output pixel of the conv
    # abs((convResult + 1 / centerPixelValue + 1) - 1)
    







