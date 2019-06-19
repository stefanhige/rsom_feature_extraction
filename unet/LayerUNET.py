# class for one CNN experiment

import torch

from torch import nn

import torch.nn.functional as F

import numpy as np

import os
import copy
import json
import warnings

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

from unet import UNet
import lossfunctions as lfs
# import nibabel as nib
from timeit import default_timer as timer

from dataloader_dev import RSOMLayerDataset
from dataloader_dev import RandomZShift, ZeroCenter, CropToEven
from dataloader_dev import DropBlue, ToTensor, precalcLossWeight



class LayerUNET():
    '''
    class for setting up, training and evaluating of layer segmentation
    with unet on RSOM dataset
    Args:
        
    '''
    def __init__(self,
                 device=torch.device('cuda'),
                 model_depth=3,
                 dataset_zshift=0,
                 dirs={'train':'','eval':'', 'model':'', 'pred':''},
                 filename = '',
                 optimizer = 'Adam',
                 initial_lr = 1e-4,
                 scheduler_patience = 3,
                 lossfn = lfs.custom_loss_1,
                 epochs = 30
                 ):
        
        # PROCESS LOGGING
        self.filename = filename
        try:
            self.logfile = open(os.path.join(dirs['model'], 'log_' + filename), 'x')
        except:
            self.logfile = open(os.path.join(dirs['model'], 'log_' + filename), 'a')
            warnings.warn('logfile already exists! appending to existing file..', UserWarning) 
        
        # MODEL
        self.model = UNet(in_channels=2,
             n_classes=2,
             depth=model_depth,
             wf=6,
             padding=True,
             batch_norm=True,
             up_mode='upconv').to(device)
        
        self.model_depth = model_depth
        
        # LOSSFUNCTION
        self.lossfn = lossfn
        
        
        # DIRECTORIES
        # Dictionary with entries 'train' 'eval' 'model' 'pred'
        self.dirs = dirs

        
        # DATASET
        self.train_dataset_zshift = dataset_zshift
        
        self.train_dataset = RSOMLayerDataset(self.dirs['train'],
            transform=transforms.Compose([RandomZShift(dataset_zshift),
                                          ZeroCenter(), 
                                          CropToEven(network_depth=self.model_depth),
                                          DropBlue(),
                                          ToTensor(),
                                          precalcLossWeight()]))
        
        self.train_dataloader = DataLoader(self.train_dataset,
                                           batch_size=1, 
                                           shuffle=False, 
                                           num_workers=4, 
                                           pin_memory=True)



        
        self.eval_dataset = RSOMLayerDataset(self.dirs['eval'],
            transform=transforms.Compose([RandomZShift(),
                                          ZeroCenter(), 
                                          CropToEven(network_depth=self.model_depth),
                                          DropBlue(),
                                          ToTensor(),
                                          precalcLossWeight()]))
        self.eval_dataloader = DataLoader(self.eval_dataset,
                                          batch_size=1, 
                                          shuffle=False, 
                                          num_workers=4, 
                                          pin_memory=True)
        
        
        # OPTIMIZER
        if optimizer == 'Adam':
            self.optimizer = torch.optim.Adam(
                    self.model.parameters(),
                    lr=self.initial_lr)
        
        # SCHEDULER
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, 
                mode='min', 
                factor=0.1,
                patience=scheduler_patience,
                verbose=True,
                threshold=1e-4,
                threshold_mode='rel',
                cooldown=0,
                min_lr=0,
                eps=1e-8)
        
        # HISTORY
        self.history = {
                'train':{'epoch': [], 'loss': []},
                'eval':{'epoch': [], 'loss': []}
                }
        
        # CURRENT EPOCH
        self.curr_epoch = None
        
        # ADDITIONAL ARGS
        self.args = self.helperClass()
        
        self.args.size_train = len(self.train_dataset)
        self.args.size_eval = len(self.eval_dataset)
        self.args.minibatch_size = 5
        self.args.device = device
        self.args.dtype = torch.float32
        self.args.non_blocking = True
        self.args.n_epochs = epochs
        self.args.data_dim = self.eval_dataset[0]['data'].shape
        
        
    def printConfiguration(self):
        for where in (sys.stdout, self.logfile):
            print('LayerUNET configuration:',file=where)
            print('TRAIN dataset loc:', self.dirs['train'], file=where)
            print('TRAIN dataset len:', self.args.size_train, file=where)
            print('EVAL dataset loc:', self.dirs['eval'], file=where)
            print('EVAL dataset len:', self.args.size_eval, file=where)
            print('EPOCHS:', self.args.n_epochs, file=where)
            print('DATA shape:', self.args.data_dim, file=where)
            print('OPTIMIZER missing name', file=where)
            print('initial lr:', self.initial_lr, file=where)
            print('lossfunction:', self.lossfn)
            
    def train_all_epochs(self): 
        self.best_model = copy.deepcopy(model.to('cpu').state_dict())
        self.best_loss = float('inf')
 
        print('Entering training loop..')
        for curr_epoch in range(self.args.n_epochs): 
            # in every epoch, generate iterators
            train_iterator = iter(self.train_dataloder)
            eval_iterator = iter(self.eval_dataloader)
            
            curr_lr = self.optimizer.state_dict()['param_groups'][0]['lr']
        
            if curr_epoch == 1:
                tic = timer()
            
            self.train(iterator=train_iterator, epoch=curr_epoch)
            
            if curr_epoch == 1:
                toc = timer()
                print('Training took:', toc - tic)
                tic = timer()
            
            self.eval(iterator=eval_iterator, epoch=curr_epoch)
        
            if curr_epoch == 1:
                toc = timer()
                print('Evaluation took:', toc - tic)
                
            print(torch.cuda.memory_cached()*1e-6,'MB memory used')
            # extract the average training loss of the epoch
            le_idx = self.history['train']['epoch'].index(curr_epoch)
            le_losses = self.history['train']['loss'][le_idx:]
            # divide by batch size (170) times dataset size
            train_loss = sum(le_losses) / (args.data_dim[0]*args.size_train)
            
            # extract most recent eval loss
            curr_loss = self.history['eval']['loss'][-1]
            
            # use ReduceLROnPlateau scheduler
            self.scheduler.step(curr_loss)
            
            if curr_loss < self.best_loss:
                self.best_loss = copy.deepcopy(curr_loss)
                self.best_model = copy.deepcopy(self.model.to('cpu').state_dict())
                found_nb = 'new best!'
            else:
                found_nb = ''
        
            print('Epoch {:d} of {:d}: lr={:.0e}, Lt={:.2e}, Le={:.2e}'.format(
                curr_epoch+1, args.n_epochs, curr_lr, train_loss, curr_loss), found_nb)
            print('Epoch {:d} of {:d}: lr={:.0e}, Lt={:.2e}, Le={:.2e}'.format(
                curr_epoch+1, args.n_epochs, curr_lr, train_loss, curr_loss), found_nb, file=self.logfile)
    
        print('finished. saving model')
        
    def train(self, iterator, epoch):
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
        # PARSE
        model = self.model
        optimizer = self.optimizer
        history = self.history
        lossfn = self.lossfn
        args = self.args
        
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
                
    def eval(self, epoch):
        '''
        evaluate with the validationset
        Args:   model
                iterator
                optimizer
                history
                epoch
                lossfn
                args
        '''
        # PARSE
        model = self.model
        iterator = self.eval_iterator
        history = self.history
        lossfn = self.lossfn
        args = self.args
        
        
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
                    label = batch['label'][:,idx:, :, :]
                    weight = batch['meta']['weight'][:, idx:, :, :]
         
                data = torch.squeeze(data, dim=0)
                label = torch.squeeze(label, dim=0)
                weight = torch.squeeze(weight, dim=0)
                
                prediction = model(data)
        
                # prediction = prediction.to('cpu')
                loss = lossfn(prediction, label, weight)
                
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
            
            self.logfile.close()
            
    def save(self):
        
        torch.save(self.best_model, os.path.join(self.dirs['model'], + 'mod_' + filename))
        
        json_f = json.dumps(self.history)
        f = open(os.path.join(self.dirs['model'],'hist_' + filename),'w')
        f.write(json_f)
        f.close()
            
            
    class helperClass():
        pass
        
        
# EXECUTION TEST
        
root_dir = '/home/gerlstefan/data/fullDataset/labeled'
train_dir = os.path.join(root_dir, 'train')
eval_dir = os.path.join(root_dir, 'val')

model_dir = '/home/gerlstefan/models/layerseg/test'
        
dirs={'train':train_dir,'eval':eval_dir, 'model':model_dir, 'pred':''}
os.environ["CUDA_VISIBLE_DEVICES"]='7'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
#net1 = LayerUNET(device=device,
#                 model_depth=3,
#                 dataset_zshift=0,
#                 dirs=dirs,
#                 filename = 'test',
#                 optimizer = 'Adam',
#                 initial_lr = 1e-4,
#                 scheduler_patience = 3,
#                 lossfn = lfs.custom_loss_1,
#                 epochs = 30
#                 )


# net1.train_all()

        
            
                
    
    