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
import nibabel as nib

import matplotlib
import matplotlib.pyplot as plt

from timeit import default_timer as timer
from datetime import date
# MY MODULES
from deep_vessel_3d import Deep_Vessel_Net_FC
from dataloader import RSOMVesselDataset
from dataloader import DropBlue, ToTensor, to_numpy
from lossfunctions import BCEWithLogitsLoss
from patch_handling import get_volume



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
                 desc,
                 sdesc,
                 device=torch.device('cuda'),
                 dirs={'train':'','eval':'', 'model':'', 'pred':''},
                 divs = (4, 4, 3),
                 offset = (6, 6, 6),
                 optimizer = 'Adam',
                 lossfn = BCEWithLogitsLoss,
                 initial_lr = 1e-6,
                 epochs=1,
                 ):

        if 'DEBUG' in globals():
            if DEBUG:
                self.DEBUG = True
            else:
                self.DEBUG = False
        else:
            self.DEBUG = False


        
        # DESCRIPTION
        self.desc = desc
        self.sdesc = sdesc

        # OUTPUT DIRECTORIES
        self.dirs = dirs
        out_root_list = os.listdir(dirs['out'])

        today = date.today().strftime('%y%m%d')
        today_existing = [el for el in out_root_list if today in el]
        if today_existing:
            nr = max([int(el[7:9]) for el in today_existing]) + 1
        else:
            nr = 0
        
        self.today_id = today + '-{:02d}'.format(nr)
        self.dirs['out'] = os.path.join(self.dirs['out'], self.today_id)
        if self.sdesc:
            self.dirs['out'] += '-' + self.sdesc
        debug('Output directory string:', self.dirs['out'])

        if not self.DEBUG: 
            os.mkdir(self.dirs['out'])
            try:
                self.logfile = open(os.path.join(self.dirs['out'], 
                    'log' + self.today_id), 'x')
            except:
                print('Couldn\'n open logfile')
        
        self.printandlog('DESCRIPTION:', desc)

        # MODEL
        self.model = Deep_Vessel_Net_FC(in_channels=1)
        
        if self.dirs['model']:
            self.printandlog('Loading model from:', self.dirs['model'])
            try:
                self.model.load_state_dict(torch.load(self.dirs['model']))
            except:
                self.printandlog('Could not load model!') 


        self.model = self.model.to(device)
        self.model = self.model.float()
       
        # LOSSUNCTION
        self.lossfn = lossfn

        # DIVS, OFFSET
        self.divs = divs
        self.offset = offset

        # DATASET
        if self.dirs['train']:
            self.train_dataset = RSOMVesselDataset(self.dirs['train'],
                                                   divs=divs, 
                                                   offset=offset,
                                                   transform=transforms.Compose([ToTensor()]))

            self.train_dataloader = DataLoader(self.train_dataset,
                                               batch_size=1, 
                                               shuffle=True, 
                                               num_workers=4, 
                                               pin_memory=True)
        if self.dirs['eval']:
            self.eval_dataset = RSOMVesselDataset(self.dirs['eval'],
                                               divs=divs,
                                               offset=offset,
                                               transform=transforms.Compose([ToTensor()]))

            self.eval_dataloader = DataLoader(self.eval_dataset,
                                              batch_size=1, 
                                              shuffle=False, 
                                              num_workers=4,
                                              pin_memory=True)
        if self.dirs['pred']:
            self.pred_dataset = RSOMVesselDataset(self.dirs['pred'],
                                              divs=divs,
                                              offset=offset,
                                              transform=transforms.Compose([ToTensor()]))


            self.pred_dataloader = DataLoader(self.pred_dataset,
                                              batch_size=1, 
                                              shuffle=False, 
                                              num_workers=4,
                                              pin_memory=True)

            if not self.DEBUG:
                os.mkdir(os.path.join(self.dirs['out'],'prediction'))
 

        # OPTIMIZER
  
        self.initial_lr = initial_lr
        if optimizer == 'Adam':
            print('Adam with LayerUnet settings just for test')
            self.optimizer = torch.optim.Adam(self.model.parameters(),
                                              lr=self.initial_lr,
                                              weight_decay = 0)

        # SCHEDULER
        self.scheduler = None

        # HISTORY
        self.history = {'train':{'epoch': [], 'loss': []},
                        'eval':{'epoch': [], 'loss': []}}
        
        # CURRENT EPOCH
        self.curr_epoch = None
        
        # ADDITIONAL ARGS
        self.args = type('args', (object,), dict())
        
        self.args.size_train = len(self.train_dataset)
        self.args.size_eval = len(self.eval_dataset)
        if self.dirs['pred']:
            self.args.size_pred = len(self.pred_dataset)
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

            #print('data shape', data.shape)
            prediction = self.model(data)
            #print('prediction shape', prediction.shape)

            loss = self.lossfn(pred=prediction, target=label)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        
            frac_epoch = epoch + i/self.args.size_train
                
            debug('Ep:', epoch, 'fracEp:',i/self.args.size_train)
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

            debug('eval, data shape:', data.shape)
            prediction = self.model(data)
            
            loss = self.lossfn(pred=prediction, target=label)
                
            # loss running variable
            running_loss += loss.data.item()
            
            debug('Ep:', epoch, 'fracEp:',i/self.args.size_eval)
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
        torch.cuda.empty_cache()
        print('Entering training loop..')
        
        for curr_epoch in range(self.args.n_epochs): 
            
            # in every epoch, generate iterators
            train_iterator = iter(self.train_dataloader)
            eval_iterator = iter(self.eval_dataloader)
            
            curr_lr = self.optimizer.state_dict()['param_groups'][0]['lr']
        
            if curr_epoch == 1:
                tic = timer()
            
            debug(torch.cuda.memory_cached()*1e-6,'MB memory used')
            debug('calling train method')
            self.train(iterator=train_iterator, epoch=curr_epoch)
            
            debug(torch.cuda.memory_cached()*1e-6,'MB memory used')
            if curr_epoch == 1:
                toc = timer()
                print('Training took:', toc - tic)
                tic = timer()
            debug('calling eval method') 
            self.eval(iterator=eval_iterator, epoch=curr_epoch)
           

            if curr_epoch == self.args.n_epochs-1:
                print('Keeping memory cached to occupy GPU... ;)')
            else:
                torch.cuda.empty_cache()
            
            debug(torch.cuda.memory_cached()*1e-6,'MB memory used')
            
            if curr_epoch == 1:
                toc = timer()
                print('Evaluation took:', toc - tic)
                
            debug(torch.cuda.memory_cached()*1e-6,'MB memory used')
            
            # extract the average training loss of the epoch
            le_idx = self.history['train']['epoch'].index(curr_epoch)
            le_losses = self.history['train']['loss'][le_idx:]
            # divide by dataset size
            train_loss = sum(le_losses) / (self.args.size_train)
            
            # extract most recent eval loss
            curr_loss = self.history['eval']['loss'][-1]
            
            # use ReduceLROnPlateau scheduler
            # self.scheduler.step(curr_loss)
            
            if curr_loss < self.best_loss:
                self.best_loss = copy.deepcopy(curr_loss)
                self.best_model = copy.deepcopy(self.model.state_dict())
                for k, v in self.best_model.items():
                    self.best_model[k] = v.to('cpu')
                found_nb = 'new best!'
            else:
                found_nb = ''
        
            self.printandlog('Epoch {:d} of {:d}: lr={:.0e}, Lt={:.2e}, Le={:.2e}'.format(
                curr_epoch+1, self.args.n_epochs, curr_lr, train_loss, curr_loss), found_nb)
#            print('Epoch {:d} of {:d}: lr={:.0e}, Lt={:.2e}, Le={:.2e}'.format(
#                curr_epoch+1, self.args.n_epochs, curr_lr, train_loss, curr_loss), found_nb, file=self.logfile)
    
        print('Training finished...')
        print('Copying last model...')
        self.last_model = copy.deepcopy(self.model.state_dict())
        for k, v in self.last_model.items():
            self.last_model[k] = v.to('cpu')
       
        if not self.DEBUG:
            print('Closing logfile..')
            self.logfile.close()
            print('Saving loss history to .json file..')
            f = open(os.path.join(self.dirs['out'],'loss' + self.today_id + '.json'),'w')
            f.write(json.dumps(self.history))
            f.close()
                        
    def predict(self, use_best=True):
        '''
        doc string missing
        '''
        print('Predicting..')
 
        if use_best:
            print('Using best model.')
            self.model.load_state_dict(self.best_model)
        else:
            print('Using last model.')

        iterator = iter(self.pred_dataloader) 
        self.model.eval()

        prediction_stack = []
        index_stack = []
        
        for i in range(self.args.size_pred):
           
            # get the next batch of the evaluation set
            batch = next(iterator)
            
            data = batch['data'].to(
                    self.args.device,
                    self.args.dtype,
                    non_blocking=self.args.non_blocking)

            debug('prediction, data shape:', data.shape)
            prediction = self.model(data)
            prediction = prediction.detach()
            
            # convert to probabilities
            sigmoid = torch.nn.Sigmoid()
            prediction = sigmoid(prediction)
            
            # otherwise can't reconstruct.
            if i==0:
                assert batch['meta']['index'].item() == 0
             
            prediction_stack.append(prediction)
            
            index_stack.append(batch['meta']['index'].item())

            # if we got all patches
            if batch['meta']['index'] == np.prod(self.divs) - 1:
                
                debug('Reconstructing volume: index stack is:')
                debug(index_stack)

                assert len(prediction_stack) == np.prod(self.divs)
                assert index_stack == list(range(np.prod(self.divs)))
                
                patches = (torch.stack(prediction_stack)).to('cpu').numpy()
                prediction_stack = []
                index_stack = []
                
                debug('patches shape:', patches.shape)
                patches = patches.squeeze()
                debug('patches shape:', patches.shape)
                
                V = get_volume(patches, self.divs, (0,0,0))
                V = to_numpy(V, batch['meta'], Vtype='label', dimorder='torch')
                debug('reconstructed volume shape:', V.shape)

                # TODO: binary cutoff??
                debug('vessel probability min/max:', np.amin(V),'/', np.amax(V))
                
                
                V[V<0.5] = 0
                Vbool = V.astype(np.bool)

                # save to file
                if not self.DEBUG:
                    if os.path.exists(self.dirs['out']):
                        # create ../prediction directory
                        dest_dir = os.path.join(self.dirs['out'],'prediction')
                    
                        fstr = batch['meta']['filename'][0].replace('.nii.gz','')  + '_pred'
                        self.saveNII(Vbool, dest_dir, fstr)
                    else:
                        print('Couldn\'t save prediction in.')

    @staticmethod
    def saveNII(V, path, fstr):
        V = V.astype(np.uint8)
        img = nib.Nifti1Image(V, np.eye(4))
    
        fstr = fstr + '.nii.gz'
        nib.save(img, os.path.join(path, fstr))

    def plot_loss(self):
        fig, ax = plt.subplots()
        ax.plot(np.array(self.history['train']['epoch']),
                np.array(self.history['train']['loss']),
                np.array(self.history['eval']['epoch'])+0.5,
                np.array(self.history['eval']['loss']))

        ax.set_yscale('log')
        ax.set(xlabel='Epoch', ylabel='loss')
        ax.grid()

        plt.legend(('train','eval'),loc='upper right')
        
        if not self.DEBUG:
            fig.savefig(os.path.join(self.dirs['out'],
                'loss' + self.today_id + '.png'))
    
    def printandlog(self, *msg):
        print(*msg)
        try:
            print(*msg, file=self.logfile)
        except:
            pass

    def printConfiguration(self, destination='both'):
        if not self.DEBUG:
            if destination == 'stdout':
                where_ = [sys.stdout]
            elif destination == 'logfile':
                where_ = [self.logfile]
            elif destination =='both':
                where_ = [sys.stdout, self.logfile]
        else:
            where_ = [sys.stdout]

        for where in where_:
            print('VesNet configuration:',file=where)
            print(self.desc, file=where)
            print('DATA: train dataset:', self.dirs['train'], file=where)
            print('             length:', self.args.size_train, file=where)
            print('       eval dataset:', self.dirs['eval'], file=where)
            print('             length:', self.args.size_eval, file=where)
            print('       sample shape:', self.args.data_shape, file=where)
            print('               divs:', self.divs, file=where)
            print('             offset:', self.offset, file=where)


            print('EPOCHS:', self.args.n_epochs, file=where)
            print('OPTIMIZER:', self.optimizer, file=where)
            print('initial lr:', self.initial_lr, file=where)
            print('LOSS: fn', self.lossfn, file=where)
            # print('      class_weight', self.class_weight, file=where)
            # print('      smoothnes param', self.lossfn_smoothness, file=where)
            print('CNN:  ', self.model, file=where)
            # if self.model gets too long, eg unet
            #print('CNN:  ', self.model.__class__.__name__, file=where)
            if self.dirs['pred']:
                print('PRED:  pred dataset:', self.dirs['pred'], file=where)
                print('             length:', self.args.size_pred, file=where)
            #print('      depth', self.model_depth, file=where)
            #print('      dropout?', self.model_dropout, file=where)
            if self.dirs['model']:
                print('LOADING MODEL  :', self.dirs['model'], file=where)
            print('OUT:               :', self.dirs['out'], file=where)

    def save_code_status(self):
        if not self.DEBUG:
            try:
                path = os.path.join(self.dirs['out'],'git')
                os.system('git log -1 | head -n 1 > {:s}.diff'.format(path))
                os.system('echo /"\n/" >> {:s}.diff'.format(path))
                os.system('git diff >> {:s}.diff'.format(path))
            except:
                self.printandlog('Saving git diff FAILED!')


    def save_model(self,model='best'):
        if not self.DEBUG:
            if model=='best':
                save_this = self.best_model
            elif model=='last':
                save_this = self.last_model
            
            torch.save(save_this, os.path.join(self.dirs['out'],'mod' + self.today_id +'.pt'))


def debug(*msg):
    ''' debug print helper function'''
    if 'DEBUG' in globals():
        if DEBUG:
            print(*msg)

global DEBUG
# DEBUG = True

root_dir = '/home/gerlstefan/data/vesnet/synthDataset/rsom_style'


desc = 'still debugging'
sdesc = 'test'


model_dir = ''
        
os.environ["CUDA_VISIBLE_DEVICES"]='7'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


train_dir = os.path.join(root_dir, '')
eval_dir = os.path.join(root_dir, '')
out_dir = '/home/gerlstefan/data/vesnet/out'

dirs={'train': train_dir,
      'eval': eval_dir, 
      'model': model_dir, 
      'pred': eval_dir,
      'out': out_dir}

net1 = VesNET(device=device,
                     desc=desc,
                     sdesc=sdesc,
                     dirs=dirs,
                     divs=(2,2,2),
                     optimizer='Adam',
                     initial_lr=1e-4,
                     epochs=1
                     )
# output structure
# ~/data/vesnet/out/

# 190831-01-str/model_190831-01.pt
# 190831-01-str/history_190831-01.json
# 190831-01-str/log_190831-01
# 190831-01-str/loss_190831-01.png
# 190831-01-str/prediction/... .nii.gz

# save current code status
# get comment id


net1.printConfiguration()
net1.save_code_status()

net1.train_all_epochs()
net1.plot_loss()
net1.save_model()

net1.predict()




