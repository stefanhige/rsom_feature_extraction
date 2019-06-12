import torch

from torch import nn

import torch.nn.functional as F

import numpy as np

import os
import copy

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

from unet import UNet
import lossfunctions as lfs
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
            
            # move back to save memory
            # prediction = prediction.to('cpu')
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
            # prediction = prediction.to('cpu')
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
# TODO 10 Jun 19
# loss is not going down? with weighted loss fn even less.
# maybe try different optimizer??
# different loss fn?
# but first look at one or two predictions of training and testset of algorithm
# maybe there's a huge difference
# ALSO: compare how los goes down within the first epoch!
# ALSO: compare training loss and test loss of data.... read again stanford cnn


     

# root_dir = '/home/gerlstefan/data/dataloader_dev'
root_dir = '/home/gerlstefan/data/fullDataset/labeled'
train_dir = os.path.join(root_dir, 'train')
eval_dir = os.path.join(root_dir, 'val')


os.environ["CUDA_VISIBLE_DEVICES"]='7'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# print('Current GPU device:', torch.cuda.current_device()
# print('model down_path first weight at', model.down_path[0].block.state_dict()['0.weight'].device)


dataset_train = RSOMLayerDataset(train_dir,
        transform=transforms.Compose([RandomZShift(),
            ZeroCenter(), 
            CropToEven(), 
            ToTensor()]))
dataloader_train = DataLoader(dataset_train,
        batch_size=1, 
        shuffle=False, 
        num_workers=4, 
        pin_memory=True)

dataset_eval = RSOMLayerDataset(eval_dir,
        transform=transforms.Compose([RandomZShift(),
            ZeroCenter(), 
            CropToEven(), 
            ToTensor()]))
dataloader_eval = DataLoader(dataset_eval,
        batch_size=1, 
        shuffle=False, 
        num_workers=4, 
        pin_memory=True)

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
args.size_eval = len(dataset_eval)
args.minibatch_size = 5
args.device = device
args.dtype = torch.float32
args.non_blocking = True
args.n_epochs = 50
# model = debugnet()
model = UNet(in_channels=3,
             n_classes=2,
             depth=3,
             wf=6,
             padding=True,
             batch_norm=True,
             up_mode='upsample').to(args.device)

model = model.float()


initial_lr = 1e-5

optimizer = torch.optim.Adam(model.parameters(), lr=initial_lr)
# this does not worrk properly? jumping around?? wtf
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1, last_epoch=-1)

def debuglossfn(prediction, label):
    ip = torch.randn((3, 2), requires_grad=True)
    tg = torch.rand((3, 2), requires_grad=False)
    return F.binary_cross_entropy(F.sigmoid(ip), tg)

history = {
    'train':{'epoch': [], 'loss': []},
    'eval':{'epoch': [], 'loss': []}
    }

best_model = copy.deepcopy(model.state_dict())
best_loss = float('inf')
 
print('Entering training loop..')
for curr_epoch in range(args.n_epochs): 
    # in every epoch, generate iterators
    iterator_train = iter(dataloader_train)
    iterator_eval = iter(dataloader_eval)
    
    curr_lr = optimizer.state_dict()['param_groups'][0]['lr']
 
    print(torch.cuda.memory_cached()*1e-6,'MB memory used')
    train(model=model,
        iterator=iterator_train,
        optimizer=optimizer,
        history=history,
        epoch=curr_epoch,
        lossfn=lfs.cross_entropy_2d,
        args=args)

    print(torch.cuda.memory_cached()*1e-6,'MB memory used')
    eval(model=model,
        iterator=iterator_eval,
        history=history,
        epoch=curr_epoch,
        lossfn=lfs.cross_entropy_2d,
        args=args)
    scheduler.step()

    # extract most recent eval loss
    curr_loss = history['eval']['loss'][-1]
    if curr_loss < best_loss:
        best_loss = copy.deepcopy(curr_loss)
        best_model = copy.deepcopy(model.state_dict())
        found_nb = 'new best!'
    else:
        found_nb = ''

    print('Epoch {:d} of {:d}: lr={:.0e}, L={:.2e}'.format(
        curr_epoch+1, args.n_epochs, curr_lr, curr_loss), found_nb)

print('finished. saving model')

torch.save(best_model, '/home/gerlstefan/src/unet/bm_std')
