# class for one CNN experiment

import torch

from torch import nn

import torch.nn.functional as F

import numpy as np

import os
import sys
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

from dataloader_dev import SwapDim



class LayerUNET():
    '''
    class for setting up, training and evaluating of layer segmentation
    with unet on RSOM dataset
    Args:
        device             torch.device()     'cuda' 'cpu'
        model_depth        int                 unet depth
        dataset_zshift     int or (int, int)   data aug. zshift
        dirs               dict of string      use these directories
        filename           string              pattern to save output
        optimizer          string
        initial_lr         float               initial learning rate
        scheduler_patience int                 n epochs before lr reduction
        lossfn             function            custom lossfunction
        class_weight       (float, float)      class weight for classes (0, 1)
        epochs             int                 number of epochs 
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
                 lossfn_smoothness = 0,
                 class_weight = None,
                 epochs = 30,
                 dropout = False
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
             up_mode='upconv',
             dropout=dropout)
        self.model_dropout = dropout
        
        self.model = self.model.to(device)
        self.model = self.model.float()
        
        print(self.model.down_path[0].block.state_dict()['0.weight'].device)

        self.model_depth = model_depth
        
        # LOSSFUNCTION
        self.lossfn = lossfn
        if class_weight is not None:
            self.class_weight = torch.tensor(class_weight, dtype=torch.float32)
            self.class_weight = self.class_weight.to(device)
        else:
            self.class_weight = None

        self.lossfn_smoothness = lossfn_smoothness
        
        
        # DIRECTORIES
        # Dictionary with entries 'train' 'eval' 'model' 'pred'
        self.dirs = dirs

        
        # DATASET
        self.train_dataset_zshift = dataset_zshift
        
        self.train_dataset = RSOMLayerDataset(self.dirs['train'],
            transform=transforms.Compose([RandomZShift(dataset_zshift),
                                          ZeroCenter(),
                                          SwapDim(),
                                          CropToEven(network_depth=self.model_depth),
                                          DropBlue(),
                                          ToTensor(),
                                          precalcLossWeight()]))
        
        self.train_dataloader = DataLoader(self.train_dataset,
                                           batch_size=1, 
                                           shuffle=True, 
                                           num_workers=4, 
                                           pin_memory=True)



        
        self.eval_dataset = RSOMLayerDataset(self.dirs['eval'],
            transform=transforms.Compose([RandomZShift(),
                                          ZeroCenter(),
                                          SwapDim(),
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
        self.initial_lr = initial_lr
        if optimizer == 'Adam':
            self.optimizer = torch.optim.Adam(
                    self.model.parameters(),
                    lr=self.initial_lr,
                    weight_decay = 0
                    )
        
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
        
    def printConfiguration(self, destination='stdout'):
        if destination == 'stdout':
            where = sys.stdout
        elif destination == 'logfile':
            where = self.logfile
        
        print('LayerUNET configuration:',file=where)
        print('DATA: train dataset loc:', self.dirs['train'], file=where)
        print('      train dataset len:', self.args.size_train, file=where)
        print('      eval dataset loc:', self.dirs['eval'], file=where)
        print('      eval dataset len:', self.args.size_eval, file=where)
        print('      shape:', self.args.data_dim, file=where)
        print('      zshift:', self.train_dataset_zshift)
        print('EPOCHS:', self.args.n_epochs, file=where)
        print('OPTIMIZER:', self.optimizer, file=where)
        print('initial lr:', self.initial_lr, file=where)
        print('LOSS: fn', self.lossfn, file=where)
        print('      class_weight', self.class_weight, file=where)
        print('      smoothnes param', self.lossfn_smoothness, file=where)
        print('CNN:  unet', file=where)
        print('      depth', self.model_depth, file=where)
        print('      dropout?', self.model_dropout, file=where)
        print('OUT:  model:', self.dirs['model'], file=where)
        print('      pred:', self.dirs['pred'], file=where)

    def train_all_epochs(self):  
        self.best_model = copy.deepcopy(self.model.state_dict())
        for k, v in self.best_model.items():
            self.best_model[k] = v.to('cpu')
        
        self.best_loss = float('inf')
        
        print('Entering training loop..')
        for curr_epoch in range(self.args.n_epochs): 
            # in every epoch, generate iterators
            train_iterator = iter(self.train_dataloader)
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
            train_loss = sum(le_losses) / (self.args.data_dim[0]*self.args.size_train)
            
            # extract most recent eval loss
            curr_loss = self.history['eval']['loss'][-1]
            
            # use ReduceLROnPlateau scheduler
            self.scheduler.step(curr_loss)
            
            if curr_loss < self.best_loss:
                self.best_loss = copy.deepcopy(curr_loss)
                self.best_model = copy.deepcopy(self.model.state_dict())
                for k, v in self.best_model.items():
                    self.best_model[k] = v.to('cpu')
                found_nb = 'new best!'
            else:
                found_nb = ''
        
            print('Epoch {:d} of {:d}: lr={:.0e}, Lt={:.2e}, Le={:.2e}'.format(
                curr_epoch+1, self.args.n_epochs, curr_lr, train_loss, curr_loss), found_nb)
            print('Epoch {:d} of {:d}: lr={:.0e}, Lt={:.2e}, Le={:.2e}'.format(
                curr_epoch+1, self.args.n_epochs, curr_lr, train_loss, curr_loss), found_nb, file=self.logfile)
    
        print('finished. saving model')
        self.logfile.close()
    
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
        # model = self.model
        # optimizer = self.optimizer
        # history = self.history
        # lossfn = self.lossfn
        # args = self.args
        
        self.model.train()
        
        for i in range(self.args.size_train):
            # get the next batch of training data
            batch = next(iterator)
            
            # label_ = batch['label']
            # print(label_.shape)
            
            # print(label_[:,:,0,:].sum().item())
            # print(label_[:,:,-1,:].sum().item())
                    
            batch['label'] = batch['label'].to(
                    self.args.device, 
                    dtype=self.args.dtype, 
                    non_blocking=self.args.non_blocking)
            batch['data'] = batch['data'].to(
                    self.args.device,
                    self.args.dtype,
                    non_blocking=self.args.non_blocking)
            batch['meta']['weight'] = batch['meta']['weight'].to(
                    self.args.device,
                    self.args.dtype,
                    non_blocking=self.args.non_blocking)
        
        
            # divide into minibatches
            minibatches = np.arange(batch['data'].shape[1],
                    step=self.args.minibatch_size)
            for i2, idx in enumerate(minibatches): 
                if idx + self.args.minibatch_size < batch['data'].shape[1]:
                    data = batch['data'][:,
                            idx:idx+self.args.minibatch_size, :, :]
                    label = batch['label'][:,
                            idx:idx+self.args.minibatch_size, :, :]
                    weight = batch['meta']['weight'][:,
                            idx:idx+self.args.minibatch_size, :, :]
                else:
                    data = batch['data'][:, idx:, :, :]
                    label = batch['label'][:, idx:, :, :]
                    weight = batch['meta']['weight'][:, idx:, :, :]
                
         
                data = torch.squeeze(data, dim=0)
                label = torch.squeeze(label, dim=0)
                weight = torch.squeeze(weight, dim=0)
                
                prediction = self.model(data)
            
                # move back to save memory
                # prediction = prediction.to('cpu')
                loss = self.lossfn(
                        pred=prediction, 
                        target=label,
                        spatial_weight=weight,
                        class_weight=self.class_weight,
                        smoothness_weight = self.lossfn_smoothness)
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
        
                frac_epoch = epoch +\
                        i/self.args.size_train +\
                        i2/(self.args.size_train * minibatches.size)
                
                # print(epoch, i/args.size_train, i2/minibatches.size)
                self.history['train']['epoch'].append(frac_epoch)
                self.history['train']['loss'].append(loss.data.item())
                
    def eval(self, iterator, epoch):
        '''
        evaluate with the validation set
        Args:   model
                iterator
                optimizer
                history
                epoch
                lossfn
                args
        '''
        # PARSE
        # model = self.model
        # history = self.history
        # lossfn = self.lossfn
        # args = self.args
        
        
        self.model.eval()
        running_loss = 0.0
        
        for i in range(self.args.size_eval):
            # get the next batch of the testset
            
            batch = next(iterator)
            batch['label'] = batch['label'].to(
                    self.args.device, 
                    dtype=self.args.dtype, 
                    non_blocking=self.args.non_blocking)
            batch['data'] = batch['data'].to(
                    self.args.device,
                    self.args.dtype,
                    non_blocking=self.args.non_blocking)
            batch['meta']['weight'] = batch['meta']['weight'].to(
                    self.args.device,
                    self.args.dtype,
                    non_blocking=self.args.non_blocking)
        
            # divide into minibatches
            minibatches = np.arange(batch['data'].shape[1],
                    step=self.args.minibatch_size)
            for i2, idx in enumerate(minibatches):
                if idx + self.args.minibatch_size < batch['data'].shape[1]:
                    data = batch['data'][:,
                            idx:idx+self.args.minibatch_size, :, :]
                    label = batch['label'][:,
                            idx:idx+self.args.minibatch_size, :, :]
                    weight = batch['meta']['weight'][:,
                            idx:idx+self.args.minibatch_size, :, :]
                else:
                    data = batch['data'][:, idx:, :, :]
                    label = batch['label'][:,idx:, :, :]
                    weight = batch['meta']['weight'][:, idx:, :, :]
         
                data = torch.squeeze(data, dim=0)
                label = torch.squeeze(label, dim=0)
                weight = torch.squeeze(weight, dim=0)
                
                prediction = self.model(data)
        
                # prediction = prediction.to('cpu')
                loss = self.lossfn(
                        pred=prediction, 
                        target=label,
                        spatial_weight=weight,
                        class_weight=self.class_weight,
                        smoothness_weight=self.lossfn_smoothness)
                
                # loss running variable
                # TODO: check if this works
                # add value for every minibatch
                # this should scale linearly with minibatch size
                # have to verify!
                running_loss += loss.data.item()
                
                # adds up all the dice coeeficients of all samples
                # processes each slice individually
                # in the end need to divide by number of samples*number of slices per sample
                # in the end it needs to divided by the number of iterations
                # running_dice += self.dice_coeff(pred=prediction,
                #                     target=label)
        
            # running_loss adds up loss for every batch and minibatch,
            # divide by size of testset*size of each batch
            epoch_loss = running_loss / (self.args.size_eval*batch['data'].shape[1])
            self.history['eval']['epoch'].append(epoch)
            self.history['eval']['loss'].append(epoch_loss)
           
    def calc_weight_std(self, model):
        '''
        calculate the standard deviation of all weights in model_dir

        '''
        if isinstance(model, nn.Module):
            model = model.state_dict()
        
        all_values = np.array([])

        for name, values in model.items():
            if 'weight' in name:
                values = values.to('cpu').numpy()
                values = values.ravel()
                all_values = np.concatenate((all_values, values))

        stdd = np.std(all_values)
        mean = np.mean(all_values)
        print('model number of weights:', len(all_values))
        print('model weights standard deviation:', stdd)
        print('model weights mean value:        ', mean)

    def jaccard_index(pred, target):
        '''
        calculate the jaccard index per slice and return
        the sum of jaccard indices
        '''
        # TODO: implementation never used or tested

        # shapes
        # [slices, x, x]

        pred_shape = pred.shape
        print(pred.shape)

        # for every slice
        jaccard_sum = 0.0
        for slc in range(pred_shape[0]):
            pflat = pred[slc, :, :]
            tflat = target[slc, :, :]
            intersection = (pflat * tflat).sum()
            jaccard_sum += intersection/(pflat.sum() + tflat.sum())
            
        return jaccard_sum

    def save(self):
        torch.save(self.best_model, os.path.join(self.dirs['model'], 'mod_' + self.filename + '.pt'))
        
        json_f = json.dumps(self.history)
        f = open(os.path.join(self.dirs['model'],'hist_' + self.filename + '.json'),'w')
        f.write(json_f)
        f.close()
            
    class helperClass():
        pass
        
        
# EXECUTION TEST
# train_dir = '/home/gerlstefan/data/dataloader_dev'
# eval_dir = train_dir

# try 4 class weights
N = 1


root_dir = '/home/gerlstefan/data/fullDataset/labeled'


model_name = '190808_dimswap_test'


model_dir = '/home/gerlstefan/models/layerseg/dimswap'
        
os.environ["CUDA_VISIBLE_DEVICES"]='7'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

for idx in range(N):
    root_dir = root_dir

    print('current model')
    print(model_name, root_dir)
    train_dir = os.path.join(root_dir, 'train')
    eval_dir = os.path.join(root_dir, 'val')
    dirs={'train':train_dir,'eval':eval_dir, 'model':model_dir, 'pred':''}

    net1 = LayerUNET(device=device,
                         model_depth=4,
                         dataset_zshift=(-50, 200),
                         dirs=dirs,
                         filename=model_name,
                         optimizer='Adam',
                         initial_lr=1e-4,
                         scheduler_patience=3,
                         lossfn=lfs.custom_loss_1_smooth,
                         lossfn_smoothness = 50,
                         epochs=30,
                         dropout=True,
                         class_weight=(0.3, 0.7),
                         )

    net1.printConfiguration()
    net1.printConfiguration('logfile')
    print(net1.model, file=net1.logfile)

    net1.train_all_epochs()
    net1.save()


# filestrings = ['190721_unet4_dropout_no_clw', '190721_unet4_dropout_low_clw']
# for i, class_weight in enumerate((None, (0.25, 0.75))):
#     print(filestrings[i])
#     print(class_weight)

#     net1 = LayerUNET(device=device,
#                      model_depth=4,
#                      dataset_zshift=(-50, 200),
#                      dirs=dirs,
#                      filename = filestrings[i],
#                      optimizer = 'Adam',
#                      initial_lr = 1e-4,
#                      scheduler_patience = 3,
#                      lossfn = lfs.custom_loss_1,
#                      epochs = 20,
#                      dropout=True,
#                      class_weight=None
#                      )

#     net1.printConfiguration()
#     net1.printConfiguration('logfile')
#     print(net1.model, file=net1.logfile)

#     net1.train_all_epochs()
#     net1.save()

# PARAMETER SWEEPS
# class weights: None,  (weight_background, weight_Epidermis)   ~ (0.11, 0.89) class distribution is (0.89, 0.11)
# lossfunctions: standard cross entropy,  weighted cross entropy, weighted cross entropy with smoothness?
# different normalization -127...126,  -5...250, 0...255
# different data augumentation values dataset_zshift   (-50, 100) ..   (-50, 200 ?)

# different depth of unet    3, 4 , 5

            
                
    
    
