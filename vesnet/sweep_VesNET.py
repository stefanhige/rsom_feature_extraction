import os
import torch
import time

from VesNET import VesNET
from lossfunctions import BCEWithLogitsLoss, dice_loss
from deep_vessel_3d import DeepVesselNet






# do parameter sweeps

desc = []
sdesc = []
sdes = []
model = []

# run 1
desc.append(('Rsom noisy dataset. 27 samples, bce loss, standard DeepVesselNet 10ep'))
sdesc.append('nrsomf_bce_gn')

model.append(DeepVesselNet(groupnorm=True)) # default settings with group norm

# run 2
desc.append(('Rsom noisy dataset. 27 samples, bce loss, DeepVesselNet, with more params 10ep'))
sdesc.append('nrsomf_bce_gn_mp')

model.append(DeepVesselNet(in_channels=2,
                           channels = [2, 10, 20, 40, 80, 1],
                           kernels = [3, 5, 5, 3, 1],
                           depth = 5, 
                           dropout=False,
                           groupnorm=True))

print('PARAM sweep')
print('no of model params:')
print(model[0].count_parameters(), model[1].count_parameters())
for i in range(2):

    DEBUG = None
    # DEBUG = True

    root_dir = '/home/gerlstefan/data/vesnet/synthDataset/rsom_style_noisy'

    model_dir = ''
            
    os.environ["CUDA_VISIBLE_DEVICES"]='0'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_dir = os.path.join(root_dir, 'train')
    eval_dir = os.path.join(root_dir, 'eval')
    out_dir = '/home/gerlstefan/data/vesnet/out'
    pred_dir = '/home/gerlstefan/data/vesnet/synth+annotDataset/eval'

    dirs={'train': train_dir,
          'eval': eval_dir, 
          'model': model_dir, 
          'pred': pred_dir,
          'out': out_dir}

    net1 = VesNET(device=device,
                  desc=desc[i],
                  sdesc=sdesc[i],
                  dirs=dirs,
                  divs=(3,3,4),
                  model=model[i],
                  batch_size=2,
                  optimizer='Adam',
                  class_weight=10,
                  initial_lr=1e-4,
                  lossfn=BCEWithLogitsLoss,
                  epochs=10,
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

    del net1
    torch.cuda.empty_cache()
    time.sleep(5)
