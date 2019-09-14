import os
import torch


from VesNET import VesNET



DEBUG = None
# DEBUG = True

root_dir = '/home/gerlstefan/data/vesnet/annotatedDataset'
# root_dir = '/home/gerlstefan/data/vesnet/synthDataset/rsom_style_noisy_small'


desc = ('retrain model trained 50 epochs on noisy rsom data on one eval dataset')
sdesc = 'rt_test'


        
os.environ["CUDA_VISIBLE_DEVICES"]='7'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# torch.backends.cudnn.benchmark=True

train_dir = os.path.join(root_dir, 'train')
eval_dir = os.path.join(root_dir, 'eval')
out_dir = '/home/gerlstefan/data/vesnet/out'

model_dir = '/home/gerlstefan/data/vesnet/out/190914-03-nrsom_50ep_clw_1/mod190914-03.pt'
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
                     divs=(1,1,2),
                     batch_size=1,
                     optimizer='Adam',
                     class_weight=1,
                     initial_lr=1e-4,
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


