import torch

from torch import nn

import torch.nn.functional as F

import numpy as np

import os
import copy
import json

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

from unet import UNet
import lossfunctions as lfs
# import nibabel as nib
from timeit import default_timer as timer

from dataloader_dev import RSOMLayerDataset
from dataloader_dev import RandomZShift, ZeroCenter, CropToEven
from dataloader_dev import DropBlue, ToTensor, precalcLossWeight

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
        
        # label_ = batch['label']
        # print(label_.shape)
        
        # print(label_[:,:,0,:].sum().item())
        # print(label_[:,:,-1,:].sum().item())
                
        batch['label'] = batch['label'].to(
                args.device, 
                dtype=args.dtype, 
                non_blocking=args.non_blocking)
        batch['data'] = batch['data'].to(
                args.device,
                args.dtype,
                non_blocking=args.non_blocking)
        batch['meta']['weight'] = batch['meta']['weight'].to(
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
                weight = batch['meta']['weight'][:,
                        idx:idx+args.minibatch_size, :, :]
            else:
                data = batch['data'][:, idx:, :, :]
                label = batch['label'][:, idx:, :, :]
                weight = batch['meta']['weight'][:, idx:, :, :]
            
 
            data = torch.squeeze(data, dim=0)
            label = torch.squeeze(label, dim=0)
            weight = torch.squeeze(weight, dim=0)
            
            prediction = model(data)
        
            # move back to save memory
            # prediction = prediction.to('cpu')
            loss = lossfn(prediction, label, weight)
            
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


     

# train_dir = '/home/gerlstefan/data/dataloader_dev'
# eval_dir = train_dir

root_dir = '/home/gerlstefan/data/fullDataset/labeled'
train_dir = os.path.join(root_dir, 'train')
eval_dir = os.path.join(root_dir, 'val')

save_path = '/home/gerlstefan/models/layerseg/test'
save_name = 'model_20190716_2__1'




logfile = open(os.path.join(save_path, 'log_' + save_name),'x')

os.environ["CUDA_VISIBLE_DEVICES"]='7'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print('Current GPU device:', torch.cuda.current_device())

zshift = (-50, 100)
print('zshift:', zshift, file=logfile)

dataset_train = RSOMLayerDataset(train_dir,
    transform=transforms.Compose([RandomZShift(zshift),
            ZeroCenter(), 
            CropToEven(),
            DropBlue(),
            ToTensor(),
            precalcLossWeight()]))
dataloader_train = DataLoader(dataset_train,
        batch_size=1, 
        shuffle=False, 
        num_workers=4, 
        pin_memory=True)

dataset_eval = RSOMLayerDataset(eval_dir,
        transform=transforms.Compose([RandomZShift(),
            ZeroCenter(), 
            CropToEven(),
            DropBlue(),
            ToTensor(),
            precalcLossWeight()]))
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
print('TRAIN dataset len', args.size_train)
print('EVAL dataset len ', args.size_eval)
args.minibatch_size = 5
args.device = device
args.dtype = torch.float32
args.non_blocking = True
args.n_epochs = 5 
args.data_dim = dataset_eval[0]['data'].shape
# model = debugnet()
model = UNet(in_channels=2,
             n_classes=2,
             depth=3,
             wf=6,
             padding=True,
             batch_norm=True,
             up_mode='upconv').to(args.device)

model = model.float()


initial_lr = 1e-4

optimizer = torch.optim.Adam(model.parameters(), lr=initial_lr)
# this does not worrk properly? jumping around?? wtf
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1, last_epoch=-1)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
        mode='min', 
        factor=0.1,
        patience=3,
        verbose=True,
        threshold=1e-4,
        threshold_mode='rel',
        cooldown=0,
        min_lr=0,
        eps=1e-8)

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

    if curr_epoch == 1:
        tic = timer()
    
    train(model=model,
        iterator=iterator_train,
        optimizer=optimizer,
        history=history,
        epoch=curr_epoch,
        lossfn=lfs.custom_loss_1,
        args=args)
    
    if curr_epoch == 1:
        toc = timer()
        print('Training took:', toc - tic)
        tic = timer()
    
    eval(model=model,
        iterator=iterator_eval,
        history=history,
        epoch=curr_epoch,
        lossfn=lfs.custom_loss_1,
        args=args)

    if curr_epoch == 1:
        toc = timer()
        print('Evaluation took:', toc - tic)


    print(torch.cuda.memory_cached()*1e-6,'MB memory used')
    # extract the average training loss of the epoch
    le_idx = history['train']['epoch'].index(curr_epoch)
    le_losses = history['train']['loss'][le_idx:]
    # divide by batch size (170) times dataset size
    train_loss = sum(le_losses) / (args.data_dim[0]*args.size_train)
    
    # extract most recent eval loss
    curr_loss = history['eval']['loss'][-1]
    
    # use ReduceLROnPlateau scheduler
    scheduler.step(curr_loss)
    
    if curr_loss < best_loss:
        best_loss = copy.deepcopy(curr_loss)
        best_model = copy.deepcopy(model.state_dict())
        found_nb = 'new best!'
    else:
        found_nb = ''

    print('Epoch {:d} of {:d}: lr={:.0e}, Lt={:.2e}, Le={:.2e}'.format(
        curr_epoch+1, args.n_epochs, curr_lr, train_loss, curr_loss), found_nb)
    print('Epoch {:d} of {:d}: lr={:.0e}, Lt={:.2e}, Le={:.2e}'.format(
        curr_epoch+1, args.n_epochs, curr_lr, train_loss, curr_loss), found_nb, file=logfile)
    

print('finished. saving model')

# save model state_dict
torch.save(best_model,os.path.join(save_path, save_name))

logfile.close()
# save history tracking
json_f = json.dumps(history)
f = open(os.path.join(save_path,'hist_' + save_name),'w')
f.write(json_f)
f.close()

