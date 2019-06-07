
import torch
from torch import nn
import torch.nn.functional as F

import numpy as np

import os

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from SizeEstimator import SizeEstimator

# import nibabel as nib
# from timeit import default_timer as timer

from dataloader_dev import RSOMLayerDataset, RandomZShift, ZeroCenter, CropToEven, ToTensor

def train(model, iterator, optimizer, history, epoch, lossfn, args):
    '''
    train one epoch
    Args:   model
            iterator
            optimizer
            history
            epoch
            lossfn
            args 
    '''
    model.train()
    for i in range(args.size_train):
        # get the next batch of training data
        batch = next(iterator)
        
        batch['label'] = batch['label'].to(
                args.device, 
                dtype=args.dtype, 
                non_blocking=args.non_blocking)
        batch['data'] = batch['data'].to(
                args.device,
                args.dtype,
                non_blocking=args.non_blocking)

        # divide into minibatches
        minibatches = np.arange(batch['data'].shape[1],
                step=args.minibatch_size)
        for i2, idx in enumerate(minibatches): 
            if idx + args.minibatch_size < batch['data'].shape[1]:
                data = batch['data'][:,
                        idx:idx+args.minibatch_size, :, :]
                label = batch['label'][:,
                        idx:idx+args.minibatch_size, :, :]
            else:
                data = batch['data'][:, idx:, :, :]
                label = batch['label'][:,idx:, :, :]
            
 
            data = torch.squeeze(data, dim=0)
            label = torch.squeeze(label, dim=0)
            prediction = model(data)
            
            loss = lossfn(prediction, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            frac_epoch = epoch +\
                    i/args.size_train +\
                    i2/(args.size_train * minibatches.size)
            
            # print(epoch, i/args.size_train, i2/minibatches.size)
            history['train']['epoch'].append(frac_epoch)
            history['train']['loss'].append(loss.data.item())


def eval(model, iterator, history, epoch, lossfn, args):
    '''
    evaluate with the testset
    Args:   model
            iterator
            optimizer
            history
            epoch
            lossfn
            args
    '''
    model.eval()
    running_loss = 0.0

    for i in range(args.size_eval):
        # get the next batch of the testset
        batch = next(iterator)
        batch['label'] = batch['label'].to(
                args.device, 
                dtype=args.dtype, 
                non_blocking=args.non_blocking)
        batch['data'] = batch['data'].to(
                args.device,
                args.dtype,
                non_blocking=args.non_blocking)
        
        # divide into minibatches
        minibatches = np.arange(batch['data'].shape[1],
                step=args.minibatch_size)
        for i2, idx in enumerate(minibatches):
            if idx + args.minibatch_size < batch['data'].shape[1]:
                data = batch['data'][:,
                        idx:idx+args.minibatch_size, :, :]
                label = batch['label'][:,
                        idx:idx+args.minibatch_size, :, :]
            else:
                data = batch['data'][:, idx:, :, :]
                label = batch['label'][:,idx:, :, :]
            
 
            data = torch.squeeze(data, dim=0)
            label = torch.squeeze(label, dim=0)
            prediction = model(data)
            
            loss = lossfn(prediction, label)
            
            # loss running variable
            # TODO: check if this works
            # add value for every minibatch
            # this should scale linearly with minibatch size
            # have to verify!
            running_loss += loss.data.item()
        
            # print(epoch, i/args.size_train, i2/minibatches.size)
    
    # running_loss adds up loss for every batch and minibatch,
    # divide by size of testset*size of each batch
    epoch_loss = running_loss / (args.size_eval*batch['data'].shape[1])
    history['eval']['epoch'].append(epoch)
    history['eval']['loss'].append(epoch_loss)


   

# root_dir = '/home/gerlstefan/data/dataloader_dev'
root_dir = '/home/gerlstefan/data/fullDataset/labeled'

os.environ["CUDA_VISIBLE_DEVICES"]=''

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')
# model = UNet(in_channels=3,
#              n_classes=2,
#              depth=3,
#              wf=6,
#              padding=True,
#              batch_norm=True,
#              up_mode='upsample').to(device)

# model = model.float()

# print('Current GPU device:', torch.cuda.current_device()
# print('model down_path first weight at', model.down_path[0].block.state_dict()['0.weight'].device)


dataset_train = RSOMLayerDataset(root_dir, 
        transform=transforms.Compose([RandomZShift(),
            ZeroCenter(), 
            CropToEven(), 
            ToTensor()]))
dataloader = DataLoader(dataset_train,
        batch_size=1, 
        shuffle=False, 
        num_workers=4, 
        pin_memory=True)

iterator_train = iter(dataloader)
iterator_eval = iter(dataloader)
class debugnet(nn.Module):
    def __init__(self):
        super(debugnet, self).__init__()
        self.conv = nn.Conv2d(1, 1, 1)
    def forward(self, x):
        x = x*2
        return x

class arg_class():
    pass

args = arg_class()

args.size_train = len(dataset_train)
args.size_eval = len(dataset_train)
args.minibatch_size = 10
args.device = device
args.dtype = torch.float32
args.non_blocking = True
args.n_epochs = 10
debugmodel = debugnet()

optimizer = torch.optim.Adam(debugmodel.parameters(), lr=1e-2)
def lossfn(prediction, label):
    ip = torch.randn((3, 2), requires_grad=True)
    tg = torch.rand((3, 2), requires_grad=False)
    return F.binary_cross_entropy(F.sigmoid(ip), tg)
history = {
    'train':{'epoch': [], 'loss': []},
    'eval':{'epoch': [], 'loss': []}
    }

for curr_epoch in range(args.n_epochs)
    train(model=debugmodel,
        iterator=iterator_train,
        optimizer=optimizer,
        history=history,
        epoch=curr_epoch,
        lossfn=lossfn,
        args=args)

    eval(model=debugmodel,
        iterator=iterator_eval,
        history=history,
        epoch=curr_epoch,
        lossfn=lossfn,
        args=args)

