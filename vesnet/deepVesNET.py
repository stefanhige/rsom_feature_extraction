import os
import torch


from VesNET import VesNET
from lossfunctions import dice_loss
from deep_vessel_3d import DeepVesselNet
# try vesnet with depth 6


DEBUG = None
# DEBUG = True

root_dir = '/home/gerlstefan/data/vesnet/annot_test_retrain_capability/'


desc = ('test train capability with deep deep ves net')
sdesc = 'idendity_from_scratch_hq0001_d6'


model_dir = ''
        
os.environ["CUDA_VISIBLE_DEVICES"]='0'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


train_dir = os.path.join(root_dir, 'train')
eval_dir = os.path.join(root_dir, 'eval')
out_dir = '/home/gerlstefan/data/vesnet/out'
pred_dir = '/home/gerlstefan/data/vesnet/annot_test_retrain_capability/eval'

dirs={'train': train_dir,
      'eval': eval_dir, 
      'model': model_dir, 
      'pred': pred_dir,
      'out': out_dir}


model_d6 = DeepVesselNet(channels = [2, 5, 10, 20, 40, 80, 1], 
                         kernels=[3, 5, 5, 3, 3, 1],
                         depth=6)
print(model_d6.count_parameters())

net1 = VesNET(device=device,
              desc=desc,
              sdesc=sdesc,
              dirs=dirs,
              divs=(2,1,2),
              offset=(7,7,7),
              model=model_d6,
              batch_size=1,
              optimizer='Adam',
              class_weight=None,
              initial_lr=1e-3,
              lossfn=dice_loss,
              epochs=100,
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




