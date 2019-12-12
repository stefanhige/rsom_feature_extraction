# only predict with trained model.
import os
import torch


from VesNET import debug
from VesNET import VesNET
from deep_vessel_3d import DeepVesselNet

DEBUG = None
# DEBUG = True

pred_dir = '~/data/vesnet/synth+annotDataset/eval'
# pred_dir = '~/data/rand'
# pred_dir = '~/data/vesnet/myskin'

desc = ('predict only test')
sdesc = 'final_VN+GN_predadj'

# model_dir = ''
model_dir = '~/data/vesnet/out/final/191211-01-final_VN+GN/mod191211-01.pt'
        
os.environ["CUDA_VISIBLE_DEVICES"]='0'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


out_dir = '~/data/vesnet/out/final'
# out_dir = '~/data/rand'
# out_dir = '~/data/layerunet/for_vesnet/selection1/vessels/'

dirs={'train': '',
      'eval': '', 
      'model': model_dir, 
      'pred': pred_dir,
      'out': out_dir}

dirs = {k: os.path.expanduser(v) for k, v in dirs.items()}

# model = DeepVesselNet() 
model = DeepVesselNet(groupnorm=True) 

# model = DeepVesselNet(in_channels=2,
#                       channels = [2, 10, 20, 40, 80, 1],
#                       kernels = [3, 5, 5, 3, 1],
#                       depth = 5, 
#                       dropout=False,
#                       groupnorm=True)

net1 = VesNET(device=device,
                     desc=desc,
                     sdesc=sdesc,
                     dirs=dirs,
                     divs=(1,1,2),
                     model=model,
                     batch_size=1,
                     ves_probability=0.92,
                     _DEBUG=DEBUG)

net1.save_code_status()

# net1.predict(use_best=False, metrics=True, adj_cutoff=True)
net1.predict_adj()

