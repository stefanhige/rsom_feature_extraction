import os
import torch


from VesNET import VesNET
from lossfunctions import BCEWithLogitsLoss
from deep_vessel_3d import DeepVesselNet


DEBUG = None
# DEBUG = True

root_dir = '~/data/vesnet/synthDataset/rsom_style_noisy_small'

desc = ('train and retrain')
sdesc = 't+rt'

model_dir = ''
        
os.environ["CUDA_VISIBLE_DEVICES"]='6'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_dir = os.path.join(root_dir, 'train')
eval_dir = os.path.join(root_dir, 'eval')
out_dir = '/home/gerlstefan/data/vesnet/out'
pred_dir = '/home/gerlstefan/data/vesnet/annotatedDataset/eval'

dirs={'train': train_dir,
      'eval': eval_dir, 
      'model': model_dir, 
      'pred': pred_dir,
      'out': out_dir}

dirs = {k: os.path.expanduser(v) for k, v in dirs.items()}

model = DeepVesselNet(groupnorm=False,
                      use_vblock=True,
                      vblock_layer=2) # default settings with group norm

# model = DeepVesselNet(in_channels=2,
#                   channels = [2, 10, 20, 40, 80, 1],
#                   kernels = [3, 5, 5, 3, 1],
#                   depth = 5, 
#                   dropout=False,
#                   groupnorm=False,
#                   use_vblock=True,
#                   vblock_layer=2)

net1 = VesNET(device=device,
              desc=desc,
              sdesc=sdesc,
              model=model,
              dirs=dirs,
              divs=(3,3,4),
              batch_size=5,
              optimizer='Adam',
              class_weight=10,
              initial_lr=1e-4,
              lossfn=BCEWithLogitsLoss,
              epochs=1,
              ves_probability=0.95,
              _DEBUG=DEBUG
              )

# CURRENT STATE
print(net1.model.count_parameters())

net1.printConfiguration()
net1.save_code_status()

net1.train_all_epochs(cleanup=False)

net1.plot_loss()
net1.save_model(pat='t')

net1.predict(cleanup=False)

if 1:
    # set up new variables for retraining
    net1.model.load_state_dict(net1.best_model)
    
    root_dir = '~/data/vesnet/synth+annot+backgDataset'
    train_dir = os.path.join(root_dir, 'train')
    eval_dir = os.path.join(root_dir, 'eval')
    pred_dir = '~/data/vesnet/synth+annot+backgDataset/eval'

    net1.dirs['train'] = os.path.expanduser(train_dir)
    net1.dirs['eval'] = os.path.expanduser(eval_dir)
    net1.dirs['pred'] = os.path.expanduser(pred_dir)

    net1.divs=(3,4,4)
    net1.args.n_epochs = 2
    net1.batch_size = 1
    net1._setup_dataloaders()
    net1.printConfiguration()
    net1.train_all_epochs()
    net1.plot_loss()
    net1.save_model()
    net1.predict()
    
