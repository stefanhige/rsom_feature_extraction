# main class for VesNET
# Stefan Gerl
#
#
#
import torch

import torch.nn.functional as F

import numpy as np

import os 
import sys 
import copy 
import json 
import warnings



class VesNET():
    '''
    class for setting up, training of vessel segmentation with deep vessel net 3d on RSOM dataset
    Args:
        device              torch.device()              'cuda' 'cpu'



        to be determined
    '''
    def __init__(self,
                 device=torch.device('cuda')
                 ):

        # MODEL
        self.model = Deep_Vessel_Net_FC(in_channels=2)

        self.model = self.model.to(device)
        self.model = self.model.float()
       
        # LOSSUNCTION
        self.lossfn = None

        # DATASET
        self.train_dataset = None

        self.train_dataloader = None

        self.eval_dataset = None

        self.eval_dataloader = None

        # OPTIMIZER
        self.optimizer = None

        # SCHEDULER
        self.scheduler = None

        # HISTORY
        self.history = {
                'train':{'epoch': [], 'loss': []},
                'eval':{'epoch': [], 'loss': []}
                  }
        
        # CURRENT EPOCH
        self.curr_epoch = None


