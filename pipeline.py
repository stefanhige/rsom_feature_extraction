
# process raw files


import os
import numpy as np
from scipy import ndimage
import nibabel as nib
import copy
import torch
from torch import nn
import torch.nn.functional as F
import shutil

from prep.classes import RSOM, RSOM_vessel
from utils.get_unique_filepath import get_unique_filepath

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

from laynet import LayerNetBase

from vesnet.deep_vessel_3d import DeepVesselNet
from vesnet.VesNET import VesNetBase, debug

from visualization.vessels.mip_label_overlay import mip_label_overlay, mip_label_overlay1
from visualization.vessels.mip_label_overlay import RsomVisualization
# from visualization.vessels.mip_label_overlay import get_unique_filepath

# define folder
def vessel_pipeline(dirs={'input':'',
                          'output':'',
                          'laynet_model':'',
                          'vesnet_model':''},
                    laynet_depth=4,
                    vesnet_model=DeepVesselNet(),
                    divs=(1,1,1),
                    ves_probability=0.9,
                    pattern=None,  #if list, use patterns, otherwise, use whole dir
                    delete_tmp=False,
                    return_img=False,
                    mip_overlay_axis=None):


    # dirs['input'] = '/home/stefan/Documents/RSOM/Diabetes/new_data/mat'
    # dirs['laynet_model'] = '/home/stefan/models/layerseg/test/mod_190731_depth4.pt'
    # dirs['vesnet_model'] = '/home/stefan/data/vesnet/out/191017-00-rt_+backg_bce_gn/mod191017-00.pt'
    # dirs['output'] = '/home/stefan/Documents/RSOM/Diabetes/new_data/out'


    tmp_layerseg_prep = os.path.join(dirs['output'], 'tmp', 'layerseg_prep')
    tmp_layerseg_out = os.path.join(dirs['output'], 'tmp', 'layerseg_out')
    tmp_vesselseg_prep = os.path.join(dirs['output'], 'tmp', 'vesselseg_prep')
    tmp_vesselseg_out = os.path.join(dirs['output'], 'tmp', 'vesselseg_out')
    tmp_vesselseg_prob = os.path.join(dirs['output'], 'tmp', 'vesselseg_out_prob')

    if not os.path.isdir(os.path.join(dirs['output'], 'tmp')):
        os.mkdir(os.path.join(dirs['output'], 'tmp'))
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

    if not os.path.isdir(tmp_vesselseg_prob):
        os.mkdir(tmp_vesselseg_prob)
    else:
        print(tmp_vesselseg_prob, 'does exist!')
    
    # mode
    print(pattern)
    if pattern == None:
        cwd = os.getcwd()
        os.chdir(dirs['input'])
        all_files = os.listdir()
        os.chdir(cwd)
    else:
        if isinstance(pattern, str):
            pattern = [pattern]
        all_files = [os.path.basename(get_unique_filepath(dirs['input'], pat)[0]) for pat in pattern]

    # ***** PREPROCESSING FOR LAYER SEGMENTATION *****
    filenameLF_LIST = [el for el in all_files if el[-6:] == 'LF.mat']

    for idx, filenameLF in enumerate(filenameLF_LIST):
        # the other ones will be automatically defined
        filenameHF = filenameLF.replace('LF.mat','HF.mat')
        
        # extract datetime
        idx_1 = filenameLF.find('_')
        idx_2 = filenameLF.find('_', idx_1+1)
        filenameSurf = 'Surf' + filenameLF[idx_1:idx_2+1] + '.mat'
        
        # merge paths
        fullpathHF = os.path.join(dirs['input'], filenameHF)
        fullpathLF = os.path.join(dirs['input'], filenameLF)
        fullpathSurf = os.path.join(dirs['input'], filenameSurf)
        
        Obj = RSOM(fullpathLF, fullpathHF, fullpathSurf)

        Obj.prepare()

        # Obj.readMATLAB()
        # Obj.flatSURFACE()
        # Obj.cutDEPTH()
        
        # # VOLUME
        # Obj.normINTENSITY()
        # Obj.rescaleINTENSITY(dynamic_rescale = False)
        # Obj.mergeVOLUME_RGB()
        Obj.save_volume(tmp_layerseg_prep, fstr = 'rgb')
        print('Processing file', idx+1, 'of', len(filenameLF_LIST))

    # ***** LAYER SEGMENTATION *****
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    LayerNetInstance = LayerNetBase(
            dirs={'model': dirs['laynet_model'],
                  'pred': tmp_layerseg_prep,
                  'out': tmp_layerseg_out},
            model_depth=laynet_depth,
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
        fullpathHF = os.path.join(dirs['input'], filenameHF)
        fullpathLF = os.path.join(dirs['input'], filenameLF)
        fullpathSurf = os.path.join(dirs['input'], filenameSurf)
        
        Obj = RSOM_vessel(fullpathLF, fullpathHF, fullpathSurf)
        
        Obj.prepare(tmp_layerseg_out, mode='pred', fstr='pred.nii.gz')
# Obj.readMATLAB()
        # Obj.flatSURFACE()
        # Obj.cutDEPTH()
        # # cut epidermis
        # Obj.cutLAYER(tmp_layerseg_out, mode='pred', fstr='pred.nii.gz')
        # # VOLUME
        # Obj.normINTENSITY()
        # Obj.rescaleINTENSITY()
        # Obj.mergeVOLUME_RGB()
        Obj.save_volume(tmp_vesselseg_prep, fstr = 'v_rgb')
        
        print('Processing file', idx+1, 'of', len(filenameLF_LIST))
        
    # ***** VESSEL SEGMENTATION *****
    _dirs={'train': '',
          'eval': '', 
          'model': dirs['vesnet_model'], 
          'pred': tmp_vesselseg_prep,
          'out': tmp_vesselseg_out}

    net1 = VesNetBase(device=device,
                         dirs=_dirs,
                         divs= divs,
                         model=vesnet_model,
                         ves_probability=ves_probability)

    net1.predict(use_best=False, 
                 metrics=False,
                 adj_cutoff=False,
                 save_ppred=True)

    files = [f for f in os.listdir(tmp_vesselseg_out) if 'ppred' in f]
    
    for f in files:
        shutil.move(os.path.join(tmp_vesselseg_out, f), 
                    os.path.join(tmp_vesselseg_prob, f))

    # ***** VISUALIZATION *****
    _dirs = {'in': dirs['input'],
            'layer': tmp_layerseg_out,
            'vessel': tmp_vesselseg_out,
            'out': dirs['output'] }
    
    if not return_img:
        mip_label_overlay(None, _dirs, plot_epidermis=True)
    elif len(pattern) == 1:
        img = mip_label_overlay1(pattern[0], _dirs, axis=mip_overlay_axis, return_img=True)
    else:
        raise NotImplementedError('can return images only for one file')
    
    if delete_tmp:
        shutil.rmtree(os.path.join(dirs['output'],'tmp'))

    if return_img:
        return img

if __name__ == '__main__':

    dirs = {'input': '~/data/pipeline/selection1/mat',
            #'laynet_model': '~/models/layerseg/test/mod_190731_depth4.pt',
            'laynet_model': '~/models/layerseg/test/mod_191101_depth5.pt',
            'vesnet_model': '~/data/vesnet/out/final/191211-01-final_VN+GN!/mod191211-01.pt',
            'output': '~/data/pipeline/selection1/refactoring-test'}

    dirs = {k: os.path.expanduser(v) for k, v in dirs.items()}


    # model = DeepVesselNet(groupnorm=False)
    model = DeepVesselNet(groupnorm=True)

    # model = DeepVesselNet(in_channels=2,
    #                       channels = [2, 10, 20, 40, 80, 1],
    #                       kernels = [3, 5, 5, 3, 1],
    #                       depth = 5, 
    #                       dropout=False,
    #                       groupnorm=True,
                          # save_memory=True)

    os.environ["CUDA_VISIBLE_DEVICES"]='3'
    
    img = vessel_pipeline(dirs=dirs,
                    laynet_depth=5,
                    vesnet_model=model,
                    ves_probability=0.96973,
                     pattern=['R_20170828155546'],  #if list, use patterns, otherwise, use whole dir
                    divs=(1,1,2),
                    delete_tmp=False,
                    return_img=False)
