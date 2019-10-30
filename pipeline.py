
# process raw files


import os
import numpy as np
from scipy import ndimage
import nibabel as nib
import copy
import torch
from torch import nn
import torch.nn.functional as F

from prep.classes import RSOM
from prep.utils.get_unique_filepath import get_unique_filepath

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

from unet.unet import UNet
import unet.lossfunctions as unet_lfs
import unet.predict as unet_pred

from unet.dataloader_dev import RSOMLayerDatasetUnlabeled 
import unet.dataloader_dev as unet_dl


# define folder
origin = '/home/stefan/Documents/RSOM/Diabetes/new_data/mat'

layerseg_model = '/home/stefan/models/layerseg/test/mod_190731_depth4.pt'

destination = '/home/stefan/Documents/RSOM/Diabetes/new_data/out'

tmp_layerseg_prep = os.path.join(destination, 'tmp', 'layerseg_prep')
tmp_layerseg_out = os.path.join(destination, 'tmp', 'layerseg_out')

if not os.path.isdir(os.path.join(destination, 'tmp')):
    os.mkdir(os.path.join(destination, 'tmp'))
else:
    print('dir does exist!')

if not os.path.isdir(tmp_layerseg_prep):
    os.mkdir(tmp_layerseg_prep)
else:
    print(tmp_layerseg_prep, 'does exist!')

if not os.path.isdir(tmp_layerseg_out):
    os.mkdir(tmp_layerseg_out)
else:
    print(tmp_layerseg_out, 'does exist!')

# mode
mode = 'dir'

if mode=='dir':
    cwd = os.getcwd()
    # change directory to origin, and get a list of all files
    os.chdir(origin)
    all_files = os.listdir()
    os.chdir(cwd)
elif mode=='list':
    patterns = ['R_20170828154106_PAT026_RL01',
                'R_20170828155546_PAT027_RL01',
                'R_20170906132142_PAT040_RL01',
                'R_20170906141354_PAT042_RL01',
                'R_20171211150527_PAT057_RL01',
                'R_20171213135032_VOL009_RL02',
                'R_20180409164251_VOL015_RL02']
    all_files = [os.path.basename(get_unique_filepath(origin, pat)[0]) for pat in patterns]

# ***** PREPROCESSING FOR LAYER SEGMENTATION *****
# extract the LF.mat files,
filenameLF_LIST = [el for el in all_files if el[-6:] == 'LF.mat']

for idx, filenameLF in enumerate(filenameLF_LIST):
    # the other ones will be automatically defined
    filenameHF = filenameLF.replace('LF.mat','HF.mat')
    
    # extract datetime
    idx_1 = filenameLF.find('_')
    idx_2 = filenameLF.find('_', idx_1+1)
    filenameSurf = 'Surf' + filenameLF[idx_1:idx_2+1] + '.mat'
    
    # merge paths
    fullpathHF = os.path.join(origin, filenameHF)
    fullpathLF = os.path.join(origin, filenameLF)
    fullpathSurf = os.path.join(origin, filenameSurf)
    
    Obj = RSOM(fullpathLF, fullpathHF, fullpathSurf)
    
    Obj.readMATLAB()
    Obj.flatSURFACE()
    Obj.cutDEPTH()
    
    # VOLUME
    Obj.normINTENSITY()
    Obj.rescaleINTENSITY(dynamic_rescale = False)
    Obj.mergeVOLUME_RGB()
    Obj.saveVOLUME(tmp_layerseg_prep, fstr = 'rgb')
    print('Processing file', idx+1, 'of', len(filenameLF_LIST))

# ***** LAYER SEGMENTATION *****

dataset_pred = RSOMLayerDatasetUnlabeled(tmp_layerseg_prep,
        transform=transforms.Compose([
            unet_dl.ZeroCenter(), 
            unet_dl.CropToEven(network_depth=4),
            unet_dl.DropBlue(),
            unet_dl.ToTensor()]))

dataloader_pred = DataLoader(dataset_pred,
        batch_size=1, 
        shuffle=False, 
        num_workers=4, 
        pin_memory=True)

args = unet_pred.arg_class()
args.size_pred = len(dataset_pred)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Predicting ", args.size_pred, " Volumes.")
args.minibatch_size = 1
args.device = device
args.dtype = torch.float32
args.non_blocking = True
args.destination_dir = tmp_layerseg_out

model = UNet(in_channels=2,
             n_classes=2,
             depth=4,
             wf=6,
             padding=True,
             batch_norm=True,
             up_mode='upconv').to(args.device)


model = model.float()

model.load_state_dict(torch.load(layerseg_model))

iterator_pred = iter(dataloader_pred)

unet_pred.pred(model=model,
        iterator=iterator_pred,
        args=args)

    
# ***** PREPROCESSING FOR VESSEL SEGMENTATION *****

# ***** VESSEL SEGMENTATION *****


# ***** VISUALIZATION *****


# os.rmdir(os.path.join(destination, 'tmp'))
# os.rmdir(tmp_layerseg_prep)
# os.rmdir(tmp_layerseg_out)



























