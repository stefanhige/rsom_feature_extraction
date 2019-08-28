# main class for VesNET
# Stefan Gerl
#


# TORCH
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch.nn.functional as F


import numpy as np
import os 
import sys 
import copy 
import json 
import warnings

# MY MODULES
from deep_vessel_3d import Deep_Vessel_Net_FC
from dataloader import RSOMVesselDataset
from dataloader import DropBlue, ToTensor, to_numpy




class VesNET():
    '''
    class for setting up, training of vessel segmentation with deep vessel net 3d on RSOM dataset
    Args:
        device             torch.device()      'cuda' 'cpu'

        initial_lr         float               initial learning rate
        epochs             int                 number of epochs 
        to be determined
    '''
    def __init__(self,
                 device=torch.device('cuda'),
                 dirs={'train':'','eval':'', 'model':'', 'pred':''},
                 divs = (10, 10, 10),
                 offset = (0, 0, 0),
                 optimizer = 'Adam',
                 initial_lr = 1e-6,
                 epochs=1,
                 ):

        self.dirs = dirs
        

        # MODEL
        self.model = Deep_Vessel_Net_FC(in_channels=1)

        self.model = self.model.to(device)
        self.model = self.model.float()
       
        # LOSSUNCTION
        self.lossfn = None

        # DATASET
        self.train_dataset = RSOMVesselDataset(self.dirs['train'],
                                               divs=divs, 
                                               offset=offset,
                                               transform=transforms.Compose([ToTensor()]))

        self.train_dataloader = DataLoader(self.train_dataset,
                                           batch_size=1, 
                                           shuffle=True, 
                                           num_workers=4, 
                                           pin_memory=True)

        self.eval_dataset = RSOMVesselDataset(self.dirs['eval'],
                                transform=transforms.Compose([ToTensor()]))

        self.eval_dataloader = DataLoader(self.eval_dataset,
                                           batch_size=1, 
                                           shuffle=False, 
                                           num_workers=4, 
                                           pin_memory=True)

        # OPTIMIZER
  
        self.initial_lr = initial_lr
        if optimizer == 'Adam':
            print('Adam with LayerUnet settings just for test')
            self.optimizer = torch.optim.Adam(
                    self.model.parameters(),
                    lr=self.initial_lr,
                    weight_decay = 0
                    )

        # SCHEDULER
        self.scheduler = None

        # HISTORY
        self.history = {
                'train':{'epoch': [], 'loss': []},
                'eval':{'epoch': [], 'loss': []}
                  }
        
        # CURRENT EPOCH
        self.curr_epoch = None
        
        # ADDITIONAL ARGS
        self.args = type('args', (object,), dict())
        
        self.args.size_train = len(self.train_dataset)
        self.args.size_eval = len(self.eval_dataset)
        self.args.non_blocking = True
        self.args.device = device
        self.args.dtype = torch.float32
        self.args.n_epochs = epochs
        self.args.data_shape = self.train_dataset[0]['data'].shape
        
    def train(self, iterator, epoch):
        '''
        train one epoch
        Args:   iterator
                epoch
        '''     
        self.model.train()
        
        for i in range(self.args.size_train):
            
            # get the next batch of training data
            batch = next(iterator)
            
            data = batch['data'].to(
                    self.args.device,
                    self.args.dtype,
                    non_blocking=self.args.non_blocking)
            label = batch['label'].to(
                    self.args.device, 
                    dtype=self.args.dtype, 
                    non_blocking=self.args.non_blocking)

                
            prediction = self.model(data)
            

            loss = self.lossfn(pred=prediction, target=label)
                
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        
            frac_epoch = epoch + i/self.args.size_train
                
            print(epoch, i/self.args.size_train)
            self.history['train']['epoch'].append(frac_epoch)
            self.history['train']['loss'].append(loss.data.item())
            
    def eval(self, iterator, epoch):
        '''
        evaluate with the validation set
        Args:   iterator
                epoch
        '''
       
        self.model.eval()
        running_loss = 0.0
        
        for i in range(self.args.size_eval):
           
            # get the next batch of the evaluation set
            batch = next(iterator)
            
            data = batch['data'].to(
                    self.args.device,
                    self.args.dtype,
                    non_blocking=self.args.non_blocking)
            label = batch['label'].to(
                    self.args.device, 
                    dtype=self.args.dtype, 
                    non_blocking=self.args.non_blocking)


            loss = self.lossfn(pred=prediction, target=label)
                
            # loss running variable
            running_loss += loss.data.item()
            
        # running_loss adds up loss for every batch,
        # divide by size of testset
        epoch_loss = running_loss / (self.args.size_eval)
        self.history['eval']['epoch'].append(epoch)
        self.history['eval']['loss'].append(epoch_loss)
            

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
            # divide by dataset size
            train_loss = sum(le_losses) / (self.args.size_train)
            
            # extract most recent eval loss
            curr_loss = self.history['eval']['loss'][-1]
            
            # use ReduceLROnPlateau scheduler
            # self.scheduler.step(curr_loss)
            
#            if curr_loss < self.best_loss:
#                self.best_loss = copy.deepcopy(curr_loss)
#                self.best_model = copy.deepcopy(self.model.state_dict())
#                for k, v in self.best_model.items():
#                    self.best_model[k] = v.to('cpu')
#                found_nb = 'new best!'
#            else:
#                found_nb = ''
            found_nb = ''
        
            print('Epoch {:d} of {:d}: lr={:.0e}, Lt={:.2e}, Le={:.2e}'.format(
                curr_epoch+1, self.args.n_epochs, curr_lr, train_loss, curr_loss), found_nb)
#            print('Epoch {:d} of {:d}: lr={:.0e}, Lt={:.2e}, Le={:.2e}'.format(
#                curr_epoch+1, self.args.n_epochs, curr_lr, train_loss, curr_loss), found_nb, file=self.logfile)
    
        print('Training finished...')
        print('Copying last model...')
        self.last_model = copy.deepcopy(self.model.state_dict())
        for k, v in self.last_model.items():
            self.last_model[k] = v.to('cpu')
        
        #self.logfile.close()
        
        


root_dir = '/home/stefan/PYTHON/synthDataset/rsom_style'


model_dir = ''
        
#os.environ["CUDA_VISIBLE_DEVICES"]='7'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


train_dir = os.path.join(root_dir, '')
eval_dir = os.path.join(root_dir, '')

dirs={'train':train_dir,'eval':eval_dir, 'model':model_dir, 'pred':''}

net1 = VesNET(device=device,
                     dirs=dirs,
                     optimizer='Adam',
                     initial_lr=1e-4,
                     epochs=30
                     )

#net1.printConfiguration()
#net1.printConfiguration('logfile')
#print(net1.model, file=net1.logfile)

net1.train_all_epochs()
#net1.save()



