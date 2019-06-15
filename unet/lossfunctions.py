import torch
# import torch.nn as nn
import warnings
# from torch import nn
import copy 
import numpy as np

def cross_entropy_2d(pred, target):
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


def custom_loss_1(pred, target):
    '''
    doc string
    '''

    fn = torch.nn.CrossEntropyLoss(reduction='none')
    
    pred = pred.float()
    target = target.long()
    unred_loss = fn(pred, target)

    # TODO fix that naming mess
    label = target

    # LOSS shape [Minibatch, Z, X]
    label_shp = label.shape
    loss_shp = unred_loss.shape
    
    # print(label_shp, loss_shp)

    # we do it in numpy and then go back to torch
    # because torch.flip() doesn't work on 1D ???

    weight = copy.deepcopy(label)

    # loop over dim 0 and 2
    for yy in np.arange(label_shp[0]):
        for xx in np.arange(label_shp[2]):
            idx_nz = torch.nonzero(label[yy, :, xx])
            idx_beg = idx_nz[0].item()

            idx_end = idx_nz[-1].item()
            # weight[yy,:idx_beg,xx] = np.flip(scalingfn(idx_beg))
            # print(idx_beg, idx_end)
            
            A = scalingfn(idx_beg)
            B = scalingfn(label_shp[1] - idx_end)

            weight[yy,:idx_beg,xx] = A.unsqueeze(0).flip(1).squeeze()
            # print('A reversed', A.unsqueeze(0).flip(1).squeeze())
            # print('A', A)
            
            weight[yy,idx_end:,xx] = B
            # weight[yy,:idx_beg,xx] = np.flip(scalingfn(idx_beg))
            # weight[yy,idx_end:,xx] = scalingfn(label_shp[1] - idx_end)

    # so now, weight should be descending to 1, where label is 1,
    # and then ascend again

    # verify this by printing one slice
    # print(weight[2,:,100].shape)
    # print(weight[2,:,100])

    # multiply weight with unreduced element-wise loss
    # to get the final loss
    
    # print('weight', weight.dtype)
    # print('unred_loss', unred_loss.dtype)
    loss = weight.float() * unred_loss
    
    # import nibabel as nib
    # V = weight.to('cpu').numpy()
    # V = V.astype(np.float)
    # img = nib.Nifti1Image(V, np.eye(4))
    
    # nib.save(img, '/home/gerlstefan/weight_test.nii.gz')

    return torch.sum(loss)

def scalingfn(l):
    '''
    l is length
    '''
    # linear, starting at 1
    y = torch.arange(l) + 1
    return y


















