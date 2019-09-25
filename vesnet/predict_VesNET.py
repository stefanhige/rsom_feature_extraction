# only predict with trained model.
import os
import torch


from VesNET import debug
from VesNET import VesNET

DEBUG = None
# DEBUG = True

pred_dir = '/home/gerlstefan/data/vesnet/annotatedDataset/train'

desc = ('predict only test')
sdesc = 'nrsomfull_50ep_dice_pred'

# model_dir = '/home/gerlstefan/data/vesnet/out/190914-10-nrsomfull_50ep/mod190914-10.pt'
model_dir = '/home/gerlstefan/data/vesnet/out/190923-01-nrsomfull_50ep_dice/mod190923-01.pt'
        
os.environ["CUDA_VISIBLE_DEVICES"]='0'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


out_dir = '/home/gerlstefan/data/vesnet/out'

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
                     ves_probability=0.95,
                     _DEBUG=DEBUG)

net1.save_code_status()

net1.predict(use_best=False, metrics=True, adj_cutoff=True)



