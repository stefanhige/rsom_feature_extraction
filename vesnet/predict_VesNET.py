# only predict with trained model.
import os
import torch


from VesNET import debug
from VesNET import VesNET

DEBUG = None
# DEBUG = True

#pred_dir = '/home/gerlstefan/data/vesnet/synth+annotDataset/eval'
pred_dir = '/home/gerlstefan/data/layerunet/for_vesnet/selection1/vessels/input'

desc = ('predict only test')
sdesc = 'rt_nolabel_pred'

# model_dir = '/home/gerlstefan/data/vesnet/out/190914-10-nrsomfull_50ep/mod190914-10.pt'
model_dir = '/home/gerlstefan/data/vesnet/out/191002-03-rt_nrsomf_bce_gn_mp/mod191002-03.pt'
        
os.environ["CUDA_VISIBLE_DEVICES"]='0'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


out_dir = '/home/gerlstefan/data/vesnet/out'

out_dir = '/home/gerlstefan/data/layerunet/for_vesnet/selection1/vessels/'

dirs={'train': '',
      'eval': '', 
      'model': model_dir, 
      'pred': pred_dir,
      'out': out_dir}

net1 = VesNET(device=device,
                     desc=desc,
                     sdesc=sdesc,
                     dirs=dirs,
                     divs=(2,2,2),
                     batch_size=1,
                     ves_probability=0.925,
                     _DEBUG=DEBUG)

net1.save_code_status()

net1.predict(use_best=False, metrics=True, adj_cutoff=False)
# net1.predict_adj()


