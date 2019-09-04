import torch

from torch import nn

import torch.nn.functional as F

import numpy as np

from scipy import ndimage
# from scipy.misc import imsave 
import os
import copy

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

from unet import UNet
import lossfunctions as lfs
import nibabel as nib

from dataloader_dev import RSOMLayerDataset, RSOMLayerDatasetUnlabeled 
from dataloader_dev import RandomZShift, ZeroCenter, CropToEven, DropBlue, ToTensor

def pred(model=None, iterator=None, history=None, lossfn=None, args=None):
    '''
    evaluate with the testset
    Args:   model
            iterator
            history
            lossfn
            args
    '''
    model.eval()

    for i in range(args.size_pred):
        # get the next volume to evaluate 
        batch = next(iterator)
        
        m = batch['meta']
        print(m['dcrop']['begin'], m['dcrop']['end'], m['lcrop']['begin'], m['lcrop']['end']) 
        
        # batch['label'] = batch['label'].to(
        #         args.device, 
        #         dtype=args.dtype, 
                # non_blocking=args.non_blocking)
        batch['data'] = batch['data'].to(
                args.device,
                args.dtype,
                non_blocking=args.non_blocking)
        
        # divide into minibatches
        minibatches = np.arange(batch['data'].shape[1],
                step=args.minibatch_size)
        # init empty prediction stack
        shp = batch['data'].shape
        print('Data shape:', shp)
        # [0 x 2 x 500 x 332]
        prediction_stack = torch.zeros((0, 2, shp[3], shp[4]),
                dtype=args.dtype,
                requires_grad = False)
        prediction_stack = prediction_stack.to(args.device)
        # print(prediction_stack.shape)

        for i2, idx in enumerate(minibatches):
            if idx + args.minibatch_size < batch['data'].shape[1]:
                data = batch['data'][:,
                        idx:idx+args.minibatch_size, :, :]
                # label = batch['label'][:,
                #         idx:idx+args.minibatch_size, :, :]
            else:
                data = batch['data'][:, idx:, :, :]
                # label = batch['label'][:,idx:, :, :]
            
 
            data = torch.squeeze(data, dim=0)
            # label = torch.squeeze(label, dim=0)
            prediction = model(data)
            # prediction = prediction.to('cpu')
            # loss = lossfn(prediction, label)
            # print(prediction.shape)
            # stack prediction to get volume again
            prediction = prediction.detach() 
            prediction_stack = torch.cat((prediction_stack, prediction), dim=0) 
        
        print(prediction_stack.shape)
        prediction_stack = prediction_stack.to('cpu')
        # transform -> labels
        
        
        label = (prediction_stack[:,1,:,:] > prediction_stack[:,0,:,:]) 
        print(label.shape) 
        print(batch['meta']['filename']) 
        m = batch['meta']
        # print(m['dcrop']['begin'], m['dcrop']['end'], m['lcrop']['begin'], m['lcrop']['end']) 

        label = to_numpy(label, m)

        filename = batch['meta']['filename'][0]
        filename = filename.replace('rgb.nii.gz','')
        label = smooth(label, filename)

        print('Saving', filename)
        saveNII(label, args.destination_dir, filename + 'pred')
        if 0:
            # compare to ground truth
            label_gt = batch['label']
      
            label_gt = torch.squeeze(label_gt, dim=0)
            label_gt = to_numpy(label_gt, m)

            label_diff = (label > label_gt).astype(np.uint8)
            label_diff += 2*(label < label_gt).astype(np.uint8)
        
            # label_diff = label != label_gt

            saveNII(label_diff, args.destination_dir, filename + 'dpred')

def smooth(label, filename):
    '''
    smooth the prediction
    '''
    
    # 1. fill holes inside the label
    ldtype = label.dtype
    label = ndimage.binary_fill_holes(label).astype(ldtype)
    label_shape = label.shape
    label = np.pad(label, 2, mode='edge')
    label = ndimage.binary_closing(label, iterations=2)
    label = label[2:-2,2:-2,2:-2]
    assert label_shape == label.shape
    
    # 2. scan along z-dimension change in label 0->1 1->0
    #    if there's more than one transition each, one needs to be dropped
    #    after filling holes, we hope to be able to drop the outer one
    # 3. get 2x 2-D surface data with surface height being the index in z-direction
    
    surf_lo = np.zeros((label_shape[1], label_shape[2]))
    
    # set highest value possible (500) as default. Therefore, empty sections
    # of surf_up and surf_lo will get smoothened towards each other, and during
    # reconstructions, we won't have any weird shapes.
    surf_up = surf_lo.copy()+label_shape[0]

    for xx in np.arange(label_shape[1]):
        for yy in np.arange(label_shape[2]):
            nz = np.nonzero(label[:,xx,yy])
            
            if nz[0].size != 0:
                idx_up = nz[0][0]
                idx_lo = nz[0][-1]
                surf_up[xx,yy] = idx_up
                surf_lo[xx,yy] = idx_lo
   
    #    smooth coarse structure, eg with a 25x25 average and crop everything which is above average*factor
    #           -> hopefully spikes will be removed.
    surf_up_m = ndimage.median_filter(surf_up, size=(26, 26), mode='nearest')
    surf_lo_m = ndimage.median_filter(surf_lo, size=(26, 26), mode='nearest')
    
    for xx in np.arange(label_shape[1]):
        for yy in np.arange(label_shape[2]):
            if surf_up[xx,yy] < surf_up_m[xx,yy]:
                surf_up[xx,yy] = surf_up_m[xx,yy]
            if surf_lo[xx,yy] > surf_lo_m[xx,yy]:
                surf_lo[xx,yy] = surf_lo_m[xx,yy]

    # apply suitable kernel in order to smooth
    # smooth fine structure, eg with a 5x5 moving average
    surf_up = ndimage.uniform_filter(surf_up, size=(9, 5), mode='nearest')
    surf_lo = ndimage.uniform_filter(surf_lo, size=(9, 5), mode='nearest')

    # 5. reconstruct label
    label_rec = np.zeros(label_shape, dtype=np.uint8)
    for xx in np.arange(label_shape[1]):
        for yy in np.arange(label_shape[2]):

            label_rec[int(np.round(surf_up[xx,yy])):int(np.round(surf_lo[xx,yy])),xx,yy] = 1     

    return label_rec

def saveNII(V, path, fstr):
    V = V.astype(np.uint8)
    img = nib.Nifti1Image(V, np.eye(4))
    
    fstr = fstr + '.nii.gz'
    nib.save(img, os.path.join(path, fstr))

def to_numpy(V, meta):
    '''
    inverse function for class ToTensor() in dataloader_dev.py 
    args
        V: torch.tensor volume
        meta: batch['meta'] information

    return V as numpy.array volume
    '''
    # torch sizes X is batch size, C is Colour
    # data
    # [X x C x Z x Y] [171 x 3 x 500-crop x 333] (without crop)
    # and for the label
    # [X x Z x Y] [171 x 500 x 333]
    
    # we want to reshape to
    # numpy sizes
    # data
    # [Z x X x Y x 3] [500 x 171 x 333 x 3]
    # label
    # [Z x X x Y] [500 x 171 x 333]
    
    # here: we only need to backtransform labels
    print(V.shape)
    if not isinstance(V, np.ndarray):
        assert isinstance(V, torch.Tensor)
        V = V.numpy()
    V = V.transpose((1, 0, 2))

    # add padding, which was removed before,
    # and saved in meta['lcrop'] and meta['dcrop']

    # structure for np.pad
    # (before0, after0), (before1, after1), ..)
    
    # parse label crop
    b = (meta['lcrop']['begin']).numpy().squeeze()
    e = (meta['lcrop']['end']).numpy().squeeze()
    print('b, e')
    print(b, e)
    print(b.shape, e.shape)
    
    pad_width = ((b[0], e[0]), (b[1], e[1]), (b[2], e[2]))
    print(V.shape)
    
    V = np.pad(V, pad_width, 'edge')

    print(V.shape)
    return V

class arg_class():
    pass

args = arg_class()

os.environ["CUDA_VISIBLE_DEVICES"]='4'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# origin = '/home/gerlstefan/data/fullDataset/labeled/val'
# origin = '/home/gerlstefan/data/dataloader_dev'
origin = '/home/gerlstefan/data/layerunet/for_vesnet/input_for_layerseg'
destination ='/home/gerlstefan/data/layerunet/for_vesnet/prediction'
model_path = '/home/gerlstefan/models/layerseg/test/mod_190731_depth4.pt'

# TODO: new dataset without labels
# or optional labels to use also with evaluation set?

# create Dataset of prediction data
dataset_pred = RSOMLayerDatasetUnlabeled(origin,
        transform=transforms.Compose([
            ZeroCenter(), 
            CropToEven(network_depth=4),
            DropBlue(),
            ToTensor()]))

dataloader_pred = DataLoader(dataset_pred,
        batch_size=1, 
        shuffle=False, 
        num_workers=4, 
        pin_memory=True)


args.size_pred = len(dataset_pred)

print("Predicting ", args.size_pred, " Volumes.")
args.minibatch_size = 1
args.device = device
args.dtype = torch.float32
args.non_blocking = True
args.destination_dir = destination

model = UNet(in_channels=2,
             n_classes=2,
             depth=4,
             wf=6,
             padding=True,
             batch_norm=True,
             up_mode='upconv').to(args.device)


model = model.float()

model.load_state_dict(torch.load(model_path))


iterator_pred = iter(dataloader_pred)

pred(model=model,
        iterator=iterator_pred,
        args=args)

    
