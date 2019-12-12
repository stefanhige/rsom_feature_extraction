#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 13:58:44 2019

@author: sgerl
"""

from pathlib import Path

import os

from classes import RSOM_vessel

import sys

if __name__ == '__main__':
    sys.path.append('../')

from utils.get_unique_filepath import get_unique_filepath



# define folder


#origin = '/home/stefan/Documents/RSOM/Diabetes/allmat'
#origin = '/home/stefan/Documents/RSOM/Diabetes/new_data/mat'

origin = '/home/stefan/Documents/RSOM/Diabetes/processableDataset/mat'

origin_layer = '/home/stefan/fbserver_ssh/data/pipeline/processableDataset/results/tmp/layerseg_out'

#origin_layer = '/home/stefan/fbserver_ssh/data/layerunet/prediction/191123_depth5_selection1'
#origin_layer = '/home/stefan/fbserver_ssh/data/layerunet/prediction/191123_depth5_newdata'
# origin_layer = '/home/stefan/Documents/RSOM/Diabetes/rednoise_labels/epidermis_cutoff'
# origin = '/media/nas_ads_mwn/AG-Ntziachristos/RSOM_Data/RSOM_Diabetes/Stefan/allmat'
# origin = '/media/nas_ads_mwn/AG-Ntziachristos/RSOM_Data/RSOM_Diabetes/Stefan/'


# destination = '/media/nas_ads_mwn/AG-Ntziachristos/RSOM_Data/RSOM_Diabetes/Stefan/'
# destination = '/home/sgerl/Documents/PYTHON/TestDataset20190411/selection/other_preproccessing_tests/sliding_mip_6'

#destination = '/home/stefan/Documents/RSOM/Diabetes/test_noise_cutaway/newdata'
destination = '/home/stefan/Documents/RSOM/Diabetes/processableDataset/selection/threshold_segmentation'


# mode
mode = 'list'

if mode=='dir':
    cwd = os.getcwd()
    # change directory to origin, and get a list of all files
    os.chdir(origin)
    all_files = os.listdir()
    os.chdir(cwd)
elif mode=='list':
#    patterns = ['R_20170828154106_PAT026_RL01',
#                 'R_20180409164251_VOL015_RL02']
    #patterns = ['R_20170925155236_PAT051_RL01']
    patterns = ['R_20180409171854_VOL018_RL02',
                'R_20170830135455_PAT032_RL01',
                'R_20180409165847_VOL016_RL02',
                'R_20180409172754_VOL019_RL02',
                'R_20171127155723_VOL004_RL01',
                'R_20170911152919_PAT045_RL01',
                'R_20170911151516_PAT044_RL01',
                'R_20170809135901_PAT020_RL01',
                'R_20171211153606_PAT059_RL01',
                'R_20180409170556_VOL017_RL01',
                'R_20170906132603_PAT040_RL02',
                'R_20170809131529_PAT018_RL01',
                'R_20171113152204_PAT055_RL01',
                'R_20170906141825_PAT042_RL02',
                'R_20170809133530_PAT019_RL01',
                'R_20180409171554_VOL018_RL01',
                'R_20171211152025_PAT058_RL01',
                'R_20171211152425_PAT058_RL02',
                'R_20170807153617_PAT014_RL01',
                'R_20170807160848_PAT016_RL01'] 
    
#    patterns = ['R_20170828154106_PAT026_RL01',
#                'R_20170828155546_PAT027_RL01',
#                'R_20170906132142_PAT040_RL01',
#                'R_20170906141354_PAT042_RL01',
#                'R_20171211150527_PAT057_RL01',
#                'R_20171213135032_VOL009_RL02',
#                'R_20180409164251_VOL015_RL02']
    all_files = [os.path.basename(get_unique_filepath(origin, pat)[0]) for pat in patterns]

# extract the LF.mat files,
filenameLF_LIST = [el for el in all_files if el[-6:] == 'LF.mat']


for idx, filenameLF in enumerate(filenameLF_LIST):
    
    #if idx >= 1:
    #    break
    # the other ones will be automatically defined
    filenameHF = filenameLF.replace('LF.mat','HF.mat')
    
    # extract datetime
    idx_1 = filenameLF.find('_')
    idx_2 = filenameLF.find('_', idx_1+1)
    filenameSurf = 'Surf' + filenameLF[idx_1:idx_2+1] + '.mat'
    
    
    # merge paths
    fullpathHF = (Path(origin) / filenameHF).resolve()
    fullpathLF = (Path(origin) / filenameLF).resolve()
    fullpathSurf = (Path(origin) / filenameSurf).resolve()
    
    Obj = RSOM_vessel(fullpathLF, fullpathHF, fullpathSurf)
    
    Obj.readMATLAB()
    
    Obj.flatSURFACE()
    Obj.cutDEPTH()
    
    # surface for quick check
    # Obj.saveSURFACE((destination + ''), fstr = 'surf')
    
    # MIP image for quick check
    Obj.calcMIP(do_plot = False)
    #Obj.saveMIP(destination, fstr = 'mip')
    
    
    # MIP 3D for annotation
    # Obj.calcMIP3D(do_plot = False)
    #Obj.saveMIP3D(destination, fstr = 'mip3d')
    
    # cut epidermis
    Obj.cutLAYER(origin_layer, mode='pred', fstr='pred.nii.gz')
    #Obj.cutLAYER(origin_layer, mode='manual', fstr='manual')

    # VOLUME
    Obj.normINTENSITY()
    
    #Obj._debug_cut_empty_or_layer(dest=os.path.join(destination,'mipproj'))
    #Obj.cut_empty_or_layer()
    #Obj.cut_empty_or_layer_manual(
            #'/home/stefan/PYTHON/HQDatasetVesselAnnot/vessels_tight_volume/manual_z_values',
            #fstr='manual')
    
    Obj.rescaleINTENSITY()
    
    
    debug = Obj.thresholdSEGMENTATION()
    #Obj.mathMORPH()
    
    Obj.saveSEGMENTATION(destination, fstr='l')
    #Obj.backgroundAnnot_replaceVessel(origin_layer, 
                                      # mode='manual',
                                      # fstr='ves_cutoff')
    
    #Obj.mergeVOLUME_RGB()
    #Obj.saveVOLUME(destination, fstr = 'v_rgb')
    
    print('Processing file', idx+1, 'of', len(filenameLF_LIST))
