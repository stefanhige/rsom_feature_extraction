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
from deep_vessel_3d import DeepVesselNet
from dataloader import RSOMVesselDataset
from dataloader import DropBlue, AddDuplicateDim, ToTensor, to_numpy
from dataloader import PrecalcSkeleton, DataAugmentation
from lossfunctions import BCEWithLogitsLoss, calc_metrics, find_cutoff
from patch_handling import get_volume


class VesNET():
    '''
    class for setting up, training of vessel segmentation with deep vessel net 3d on RSOM dataset
    Args:
        device             torch.device()      'cuda' 'cpu', obsolete, as sometimes .cuda() is used
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
                 batch_size = 1,
                 optimizer = 'Adam',
                 lossfn = BCEWithLogitsLoss,
                 class_weight = None,
                 initial_lr = 1e-6,
                 epochs=1,
                 ves_probability=0.5,
                 _DEBUG=False
                 ):

        if _DEBUG:
            self.DEBUG = True
            print('DEBUG MODE')
            global DEBUG
            DEBUG = True
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
        self.model = DeepVesselNet(in_channels=2,
                                   dropout=False,
                                   batchnorm=False)
        
        if self.dirs['model']:
            self.printandlog('Loading model from:', self.dirs['model'])
            try:
                self.model.load_state_dict(torch.load(self.dirs['model']))
            except:
                self.printandlog('Could not load model!') 


        self.model = self.model.to(device)
        self.model = self.model.float()

        # VESSEL prediction probability boundary
        self.ves_probability = ves_probability
       
        # LOSSUNCTION
        self.lossfn = lossfn
        self.class_weight = class_weight

        # DIVS, OFFSET
        self.divs = divs
        self.offset = offset

        # DATASET
        self.batch_size = batch_size
        if self.dirs['train']:
            self.train_dataset = RSOMVesselDataset(self.dirs['train'],
                                                   divs=divs, 
                                                   offset=offset,
                                                   transform=transforms.Compose([
                                                       DropBlue(),
                                                       DataAugmentation(mode='rsom'),
                                                       PrecalcSkeleton(),
                                                       ToTensor()]))

            self.train_dataloader = DataLoader(self.train_dataset,
                                               batch_size=self.batch_size, 
                                               shuffle=True, 
                                               num_workers=4, 
                                               pin_memory=True)

        if self.dirs['eval']:
            self.eval_dataset = RSOMVesselDataset(self.dirs['eval'],
                                               divs=divs,
                                               offset=offset,
                                               transform=transforms.Compose([
                                                   DropBlue(),
                                                   PrecalcSkeleton(),
                                                   ToTensor()]))

            self.eval_dataloader = DataLoader(self.eval_dataset,
                                              batch_size=self.batch_size, 
                                              shuffle=False, 
                                              num_workers=4,
                                              pin_memory=True)

        if self.dirs['pred']:
            self.pred_dataset = RSOMVesselDataset(self.dirs['pred'],
                                              divs=divs,
                                              offset=offset,
                                              transform=transforms.Compose([
                                                  DropBlue(),
                                                  PrecalcSkeleton(),
                                                  ToTensor()]))


            self.pred_dataloader = DataLoader(self.pred_dataset,
                                              batch_size=1, # easier for reconstruction 
                                              shuffle=False, 
                                              num_workers=4,
                                              pin_memory=True)

            if not self.DEBUG:
                os.mkdir(os.path.join(self.dirs['out'],'prediction'))
 
        # OPTIMIZER
        self.initial_lr = initial_lr
        if optimizer == 'Adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(),
                                              lr=self.initial_lr,
                                              weight_decay = 0)
        # SCHEDULER
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, 
                mode='min', 
                factor=0.1,
                patience=2,
                verbose=True,
                threshold=1e-4,
                threshold_mode='rel',
                cooldown=0,
                min_lr=0,
                eps=1e-8)

        # HISTORY
        self.history = {'train':{'epoch': [], 'loss': [], 'unred_loss': []},
                'eval':{'epoch': [], 'loss': [], 'cl_score': [],
                    'out_score': [], 'dice': []}}
        
        # CURRENT EPOCH
        self.curr_epoch = None
        
        # ADDITIONAL ARGS
        self.args = type('args', (object,), dict())
        if self.dirs['train']: 
            self.args.size_train = len(self.train_dataset)
            self.args.size_eval = len(self.eval_dataset)
        if self.dirs['pred']:
            self.args.size_pred = len(self.pred_dataset)
        self.args.non_blocking = True
        self.args.device = device
        self.args.dtype = torch.float32
        self.args.n_epochs = epochs
        if self.dirs['train']:
            self.args.data_shape = self.train_dataset[0]['data'].shape
        else:
            self.args.data_shape = self.pred_dataset[0]['data'].shape

    def train(self, iterator, epoch):
        '''
        train one epoch
        Args:   iterator
                epoch
        '''     
        self.model.train()
        # number of iterations needed
        n_iter = int(np.ceil(self.args.size_train/self.batch_size))
        for i in range(n_iter):
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
            # debug('data shape:', data.shape)
            # debug('label shape:', label.shape)
            prediction = self.model(data)
            # debug('prediction shape:', prediction.shape)

            loss = self.lossfn(pred=prediction, target=label, weight=self.class_weight)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
           
            frac_epoch = epoch + i/n_iter
                
            curr_batch_size = data.shape[0]

            debug('Ep:', epoch, 'fracEp:', (i+1)/n_iter, 'batch:', curr_batch_size)
            self.history['train']['epoch'].append(frac_epoch)
            self.history['train']['loss'].append(loss.data.item())
            self.history['train']['unred_loss'].append(curr_batch_size * loss.data.item())
            
    def eval(self, iterator, epoch):
        '''
        evaluate with the validation set
        Args:   iterator
                epoch
        '''
       
        self.model.eval()
        running_loss = 0.0
        # running_metrics = {'cl_score': 0.0,
        #                    'out_score': 0.0,
        #                    'dice': 0.0}

        cl_score_stack = []
        out_score_stack = []
        dice_stack = []

        n_iter = int(np.ceil(self.args.size_eval/self.batch_size))
        
        for i in range(n_iter):
           
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
            
            loss = self.lossfn(pred=prediction, target=label, weight=self.class_weight)

            curr_batch_size = data.shape[0]
            
            running_loss += curr_batch_size * loss.data.item()
            # maybe this helps for memory leak?
            del loss
            
            sigmoid = torch.nn.Sigmoid()
            bool_prediction = sigmoid(prediction) >= self.ves_probability
            # metrics = calc_metrics(bool_prediction, label, batch['meta']['label_skeleton'])

            cl_score, out_score, dice = calc_metrics(bool_prediction, 
                                                     label, 
                                                     batch['meta']['label_skeleton'])

            
            cl_score_stack.append(curr_batch_size * cl_score)
            out_score_stack.append(curr_batch_size * out_score) 
            dice_stack.append(curr_batch_size * dice)
                            
            # running_metrics['cl_score'] += curr_batch_size * metrics['cl_score']
            # running_metrics['out_score'] += curr_batch_size * metrics['out_score']
            # running_metrics['dice'] += curr_batch_size * metrics['dice']
            
            debug('Ep:', epoch, 'fracEp:', (i+1)/n_iter, 'batch', curr_batch_size)
        debug('cl_score_stack:', cl_score_stack)
        epoch_cl_score = np.nansum(cl_score_stack) / self.args.size_eval
        epoch_out_score = np.nansum(out_score_stack) / self.args.size_eval
        epoch_dice = np.nansum(dice_stack) / self.args.size_eval
        
        # epoch_cl_score = running_metrics['cl_score'] / self.args.size_eval
        # epoch_out_score = running_metrics['out_score'] / self.args.size_eval
        # epoch_dice = running_metrics['dice'] / self.args.size_eval

        self.history['eval']['cl_score'].append(epoch_cl_score)
        self.history['eval']['out_score'].append(epoch_out_score)
        self.history['eval']['dice'].append(epoch_dice)

        epoch_loss = running_loss / self.args.size_eval
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
            
            debug('calling train method')
            self.train(iterator=train_iterator, epoch=curr_epoch)

            torch.cuda.empty_cache()
            
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
            
            if curr_epoch == 1:
                toc = timer()
                print('Evaluation took:', toc - tic)
                
            # extract the average training loss of the epoch
            le_idx = self.history['train']['epoch'].index(curr_epoch)
            le_losses = self.history['train']['unred_loss'][le_idx:]
            # divide by dataset size

            train_loss = sum(le_losses) / self.args.size_train
            
            # extract most recent eval loss
            curr_loss = self.history['eval']['loss'][-1]
            curr_out_score = self.history['eval']['out_score'][-1]
            curr_cl_score = self.history['eval']['cl_score'][-1]
            curr_dice = self.history['eval']['dice'][-1]
            
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
        
            self.printandlog('Epoch {:3d} of {:3d}: lr={:.0e}, Lt={:.2e}, Le={:.2e}'.format(
                curr_epoch+1, self.args.n_epochs, curr_lr, train_loss, curr_loss), found_nb)
            self.printandlog('                : cl={:.3f}, os={:.3f}, di={:.3f}'.format(
                curr_cl_score, curr_out_score, curr_dice))
    
        print('Training finished...')
        print('Copying last model...')
        self.last_model = copy.deepcopy(self.model.state_dict())
        for k, v in self.last_model.items():
            self.last_model[k] = v.to('cpu')
       
        if not self.DEBUG:
            if not self.dirs['pred']:
                print('Closing logfile..')
                self.logfile.close()
            print('Saving loss history to .json file..')
            f = open(os.path.join(self.dirs['out'],'loss' + self.today_id + '.json'),'w')
            f.write(json.dumps(self.history))
            f.close()
                        
    def predict(self, use_best=True, metrics=True, adj_cutoff=True):
        '''
        doc string missing
        '''
        print('Predicting..')
        # TODO: better solution needed?
        if use_best:
            print('Using best model.')
            self.model.load_state_dict(self.best_model)
        else:
            print('Using last model.')

        iterator = iter(self.pred_dataloader) 
        self.model.eval()

        prediction_stack = []
        index_stack = []

        if metrics:
            cl_score_stack = []
            out_score_stack = []
            dice_stack = []

        if adj_cutoff:
            label_stack = []

        
        for i in range(self.args.size_pred):
           
            # get the next batch of the evaluation set
            batch = next(iterator)
            
            data = batch['data'].to(
                    self.args.device,
                    self.args.dtype,
                    non_blocking=self.args.non_blocking)
            
            if metrics:
                label = batch['label'].to(
                         self.args.device,
                         self.args.dtype,
                         non_blocking=self.args.non_blocking)
            

            debug('prediction, data shape:', data.shape)
            prediction = self.model(data)
            prediction = prediction.detach()
            
            # convert to probabilities
            sigmoid = torch.nn.Sigmoid()
            prediction = sigmoid(prediction)

            # calculate metrics
            if metrics:
                cl_score, out_score, dice = calc_metrics(prediction >= self.ves_probability, 
                                                         label, 
                                                         batch['meta']['label_skeleton'])
            
            # otherwise can't reconstruct.
            if i==0:
                assert batch['meta']['index'].item() == 0
             
            prediction_stack.append(prediction)
            index_stack.append(batch['meta']['index'].item())
            if metrics:
                del label
            if adj_cutoff:
                label_stack.append(batch['label'].to(self.args.dtype))
            
            if metrics:
                cl_score_stack.append(cl_score)
                out_score_stack.append(out_score)
                dice_stack.append(dice)


            # if we got all patches
            if batch['meta']['index'] == np.prod(self.divs) - 1:
                
                debug('Reconstructing volume: index stack is:')
                debug(index_stack)

                assert len(prediction_stack) == np.prod(self.divs)
                assert index_stack == list(range(np.prod(self.divs)))
                
                patches = (torch.stack(prediction_stack)).to('cpu').numpy()
                prediction_stack = []
                index_stack = []

                if metrics:
                    self.printandlog('Metrics of', batch['meta']['filename'][0])
                    self.printandlog('  cl={:.3f}, os={:.3f}, di={:.3f}'.format(
                        np.nanmean(cl_score_stack), 
                        np.nanmean(out_score_stack), 
                        np.nanmean(dice_stack)))
                    print(cl_score_stack)
                    print(out_score_stack)
                    print(dice_stack)
                    
                    out_score_stack = []
                    cl_score_stack = []
                    dice_stack = []

                debug('patches shape:', patches.shape)
                patches = patches.squeeze()
                debug('patches shape:', patches.shape)
                
                V = get_volume(patches, self.divs, (0,0,0))
                V = to_numpy(V, batch['meta'], Vtype='label', dimorder='torch')
                debug('reconstructed volume shape:', V.shape)

                # TODO: binary cutoff??
                debug('vessel probability min/max:', np.amin(V),'/', np.amax(V))

                if adj_cutoff:
                    label_patches = (torch.stack(label_stack)).numpy().squeeze()
                    label_stack = []
                    L = get_volume(label_patches, self.divs, (0, 0, 0))
                    L = to_numpy(L, batch['meta'], Vtype='label', dimorder='torch')
                    id_cutoff, id_dice = find_cutoff(pred=V, label=L)
                    self.printandlog('Finding ideal p of ', batch['meta']['filename'][0])
                    self.printandlog('Result. at p={:.5f} : dice={:.5f}'.format(
                        id_cutoff, id_dice))
                    Vbool = V >= id_cutoff
                else:
                    Vbool = V >= self.ves_probability

                # save to file
                if not self.DEBUG:
                    if os.path.exists(self.dirs['out']):
                        # create ../prediction directory
                        dest_dir = os.path.join(self.dirs['out'],'prediction')
                    
                        fstr = batch['meta']['filename'][0].replace('.nii.gz','')  + '_pred'
                        self.saveNII(Vbool.astype(np.uint8), dest_dir, fstr)
                        fstr = fstr.replace('_pred', '_ppred')
                        self.saveNII(V, dest_dir, fstr)
                    else:
                        print('Couldn\'t save prediction in.')
        if not self.DEBUG:
            try:
                print('Closing logfile..')
                self.logfile.close()
            except:
                pass

    @staticmethod
    def saveNII(V, path, fstr):
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
            print('         batch size:', self.batch_size, file=where)


            print('EPOCHS:', self.args.n_epochs, file=where)
            print('OPTIMIZER:', self.optimizer, file=where)
            print('initial lr:', self.initial_lr, file=where)
            print('LOSS: fn', self.lossfn, file=where)
            print('class_weight', self.class_weight, file=where)
            # print('      smoothnes param', self.lossfn_smoothness, file=where)
            print('CNN:  ', self.model, file=where)
            # if self.model gets too long, eg unet
            #print('CNN:  ', self.model.__class__.__name__, file=where)
            if self.dirs['pred']:
                print('PRED:  pred dataset:', self.dirs['pred'], file=where)
                print('             length:', self.args.size_pred, file=where)
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

    def save_model(self, model='best'):
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

if __name__ == '__main__': 

    DEBUG = None
    DEBUG = True

    root_dir = '/home/gerlstefan/data/vesnet/synthDataset/rsom_style_small'


    desc = ('Rsom noisy dataset. 27 samples, 50 epochs')
    sdesc = 'nrsomfull_50ep'


    model_dir = ''
            
    os.environ["CUDA_VISIBLE_DEVICES"]='0'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    train_dir = os.path.join(root_dir, 'train')
    eval_dir = os.path.join(root_dir, 'eval')
    out_dir = '/home/gerlstefan/data/vesnet/out'
    pred_dir = '/home/gerlstefan/data/vesnet/annotatedDataset/eval'

    dirs={'train': train_dir,
          'eval': eval_dir, 
          'model': model_dir, 
          'pred': pred_dir,
          'out': out_dir}

    net1 = VesNET(device=device,
                         desc=desc,
                         sdesc=sdesc,
                         dirs=dirs,
                         divs=(3,3,3),
                         batch_size=2,
                         optimizer='Adam',
                         class_weight=1,
                         initial_lr=1e-4,
                         epochs=1,
                         ves_probability=0.95,
                         _DEBUG=DEBUG
                         )

    # CURRENT STATE

    net1.printConfiguration()
    net1.save_code_status()

    net1.train_all_epochs()

    net1.plot_loss()
    net1.save_model()

    net1.predict()




