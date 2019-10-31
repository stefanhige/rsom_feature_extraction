
# process raw files


import os
import numpy as np
from scipy import ndimage
import nibabel as nib
import copy
import torch
from torch import nn
import torch.nn.functional as F

from prep.classes import RSOM, RSOM_vessel
from prep.utils.get_unique_filepath import get_unique_filepath

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

# REMOVE
from laynet._model import UNet
import laynet._metrics as unet_lfs
import laynet.predict as unet_pred

from laynet import LayerNetBase

from laynet._dataset import RSOMLayerDatasetUnlabeled 
import laynet._dataset as unet_dl

from vesnet.deep_vessel_3d import DeepVesselNet
from vesnet.VesNET import VesNET, debug

from visualization.vessels.mip_label_overlay import mip_label_overlay
from visualization.vessels.mip_label_overlay import RsomVisualization
from visualization.vessels.mip_label_overlay import get_unique_filepath

# define folder
origin = '/home/stefan/Documents/RSOM/Diabetes/new_data/mat'

layerseg_model = '/home/stefan/models/layerseg/test/mod_190731_depth4.pt'
vesselseg_model = '/home/stefan/data/vesnet/out/191017-00-rt_+backg_bce_gn/mod191017-00.pt'

destination = '/home/stefan/Documents/RSOM/Diabetes/new_data/out'

tmp_layerseg_prep = os.path.join(destination, 'tmp', 'layerseg_prep')
tmp_layerseg_out = os.path.join(destination, 'tmp', 'layerseg_out')
tmp_vesselseg_prep = os.path.join(destination, 'tmp', 'vesselseg_prep')
tmp_vesselseg_out = os.path.join(destination, 'tmp', 'vesselseg_out')

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

if not os.path.isdir(tmp_vesselseg_prep):
    os.mkdir(tmp_vesselseg_prep)
else:
    print(tmp_vesselseg_prep, 'does exist!')

if not os.path.isdir(tmp_vesselseg_out):
    os.mkdir(tmp_vesselseg_out)
else:
    print(tmp_vesselseg_out, 'does exist!')

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

# TODO: HACK
filenameLF_LIST = [filenameLF_LIST[0]]

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
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

LayerNetInstance = LayerNetBase(
        dirs={'model': layerseg_model,
              'pred': tmp_layerseg_prep,
              'out': tmp_layerseg_out},
        device=device)

LayerNetInstance.predict()


    
# ***** PREPROCESSING FOR VESSEL SEGMENTATION *****

for idx, filenameLF in enumerate(filenameLF_LIST):

    filenameHF = filenameLF.replace('LF.mat','HF.mat')
    
    # extract datetime
    idx_1 = filenameLF.find('_')
    idx_2 = filenameLF.find('_', idx_1+1)
    filenameSurf = 'Surf' + filenameLF[idx_1:idx_2+1] + '.mat'
    
    # merge paths
    fullpathHF = os.path.join(origin, filenameHF)
    fullpathLF = os.path.join(origin, filenameLF)
    fullpathSurf = os.path.join(origin, filenameSurf)
    
    Obj = RSOM_vessel(fullpathLF, fullpathHF, fullpathSurf)
   
    
    Obj.readMATLAB()
    
    Obj.flatSURFACE()
    Obj.cutDEPTH()
    
    # cut epidermis
    Obj.cutLAYER(tmp_layerseg_out, mode='pred', fstr='pred.nii.gz')
    
    # VOLUME
    Obj.normINTENSITY()
    Obj.rescaleINTENSITY()
    Obj.mergeVOLUME_RGB()
    Obj.saveVOLUME(tmp_vesselseg_prep, fstr = 'v_rgb')
    
    print('Processing file', idx+1, 'of', len(filenameLF_LIST))
    
# ***** VESSEL SEGMENTATION *****

DEBUG = None
# DEBUG = True

desc = ('pipeline test')
sdesc = ''
        
# out_dir = '~/data/vesnet/out'

dirs={'train': '',
      'eval': '', 
      'model': vesselseg_model, 
      'pred': tmp_vesselseg_prep,
      'out': destination}

# model = DeepVesselNet(groupnorm=True) # default settings with group norm

model = DeepVesselNet(in_channels=2,
                      channels = [2, 10, 20, 40, 80, 1],
                      kernels = [3, 5, 5, 3, 1],
                      depth = 5, 
                      dropout=False,
                      groupnorm=True)

net1 = VesNET(device=device,
                     desc=desc,
                     sdesc=sdesc,
                     dirs=dirs,
                     divs=(4,4,3),
                     model=model,
                     batch_size=1,
                     ves_probability=0.855,
                     _DEBUG=DEBUG)

net1.save_code_status()

net1.predict(use_best=False, metrics=True, adj_cutoff=False)
# net1.predict_adj()



# ***** VISUALIZATION *****


dirs = {'in': origin,
        'layer': tmp_layerseg_out,
        'vessel': tmp_vesselseg_out,
        'out': destination }


mip_label_overlay(None, dirs, plot_epidermis=False)


# os.rmdir(os.path.join(destination, 'tmp'))
# os.rmdir(tmp_layerseg_prep)
# os.rmdir(tmp_layerseg_out)



























