import torch
# import torch.nn as nn
import warnings
# from torch import nn

def cross_entropy_2d(pred, target):
    '''
    doc string
    '''
    fn = torch.nn.CrossEntropyLoss(weight=torch.tensor([0.1, 0.9]).to('cuda'))
    
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
        target = target[:,:pd1,:pd2]

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

