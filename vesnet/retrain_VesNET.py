import os
import torch


from VesNET import VesNET
from lossfunctions import BCEWithLogitsLoss
from deep_vessel_3d import DeepVesselNet


DEBUG = None
# DEBUG = True

root_dir = '/home/gerlstefan/data/vesnet/synth+annotDataset/'
# root_dir = '/home/gerlstefan/data/vesnet/synthDataset/rsom_style_noisy_small'
# root_dir = '/home/gerlstefan/data/vesnet/annot_test_retrain_capability'

desc = ('retrain on 3 synth + 2 rsom'
        ' 190929-01-nrsomf_bce_gn_mp')
sdesc = 'rt_nrsomf_bce_gn_mp'


os.environ["CUDA_VISIBLE_DEVICES"]='0'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# torch.backends.cudnn.benchmark=True

train_dir = os.path.join(root_dir, 'train')
eval_dir = os.path.join(root_dir, 'eval')
out_dir = '/home/gerlstefan/data/vesnet/out'

model_dir = '/home/gerlstefan/data/vesnet/out/190929-02-nrsomf_bce_gn_mp/mod190929-02.pt'
# model_dir = ''
pred_dir = '/home/gerlstefan/data/vesnet/synth+annotDataset/eval'

dirs={'train': train_dir,
      'eval': eval_dir,
      'model': model_dir,
      'pred': pred_dir,
      'out': out_dir}


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
                     divs=(3,3,3),
                     model=model,
                     batch_size=1,
                     optimizer='Adam',
                     class_weight=10,
                     initial_lr=1e-4,
                     lossfn=BCEWithLogitsLoss,
                     epochs=50,
                     ves_probability=0.85,
                     _DEBUG=DEBUG
                     )

# CURRENT STATE

net1.printConfiguration()
net1.save_code_status()

net1.train_all_epochs()

net1.plot_loss()
net1.save_model()

net1.predict()


