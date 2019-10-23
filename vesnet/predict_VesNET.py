# only predict with trained model.
import os
import torch


from VesNET import debug
from VesNET import VesNET
from deep_vessel_3d import DeepVesselNet

DEBUG = None
# DEBUG = True

pred_dir = '~/data/vesnet/synth+annot+backgDataset/eval'
# pred_dir = '/home/gerlstefan/data/layerunet/for_vesnet/selection1/vessels/input'

desc = ('predict only test')
sdesc = 'rt_+backg_bce_gn'

# model_dir = '/home/gerlstefan/data/vesnet/out/190914-10-nrsomfull_50ep/mod190914-10.pt'
model_dir = '~/data/vesnet/out/191017-00-rt_+backg_bce_gn/mod191017-00.pt'
        
os.environ["CUDA_VISIBLE_DEVICES"]='0'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


out_dir = '~/data/vesnet/out'

# out_dir = '/home/gerlstefan/data/layerunet/for_vesnet/selection1/vessels/'

dirs={'train': '',
      'eval': '', 
      'model': model_dir, 
      'pred': pred_dir,
      'out': out_dir}

dirs = {k: os.path.expanduser(v) for k, v in dirs.items()}

model = DeepVesselNet(groupnorm=True) # default settings with group norm
net1 = VesNET(device=device,
                     desc=desc,
                     sdesc=sdesc,
                     dirs=dirs,
                     divs=(4,4,3),
                     model=model,
                     batch_size=1,
                     ves_probability=0.835,
                     _DEBUG=DEBUG)

net1.save_code_status()

# net1.predict(use_best=False, metrics=True, adj_cutoff=False)
net1.predict_adj()


