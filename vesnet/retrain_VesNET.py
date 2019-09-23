import os
import torch


from VesNET import VesNET
from lossfunctions import dice_loss



DEBUG = None
DEBUG = True

root_dir = '/home/gerlstefan/data/vesnet/synthDataset/rsom_style_noisy_small'
# root_dir = '/home/gerlstefan/data/vesnet/synthDataset/rsom_style_noisy_small'


desc = ('3 synth + 2 rsom , '
        'try combinded retrain')
sdesc = 'rt_test_comb'


os.environ["CUDA_VISIBLE_DEVICES"]='0'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# torch.backends.cudnn.benchmark=True

train_dir = os.path.join(root_dir, 'train')
eval_dir = os.path.join(root_dir, 'eval')
out_dir = '/home/gerlstefan/data/vesnet/out'

model_dir = '/home/gerlstefan/data/vesnet/out/190914-10-nrsomfull_50ep/mod190914-10.pt'
# model_dir = ''
pred_dir = '/home/gerlstefan/data/vesnet/annotatedDataset/eval'

dirs={'train': train_dir,
      'eval': eval_dir,
      'model': model_dir,
      'pred': pred_dir,
      'out': out_dir}

net1 = VesNET(device=device,
                     desc=desc,
                     sdesc=sdesc,
                     dirs=dirs,
                     divs=(2,2,2),
                     batch_size=1,
                     optimizer='Adam',
                     class_weight=None,
                     initial_lr=1e-4,
                     lossfn=dice_loss,
                     epochs=20,
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


