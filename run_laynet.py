
import torch
import os
from laynet._metrics import custom_loss_1_smooth
from laynet import LayerNet, LayerNetBase

mode = 'predict'

if mode == 'train':
    N = 1
    root_dir = '/home/gerlstefan/data/layerunet/fullDataset/labeled'


    model_name = '191102_depth4'

    model_dir = '/home/gerlstefan/models/layerseg/test'
            
    os.environ["CUDA_VISIBLE_DEVICES"]='6'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for idx in range(N):
        root_dir = root_dir

        print('current model')
        print(model_name, root_dir)
        train_dir = os.path.join(root_dir, 'train')
        eval_dir = os.path.join(root_dir, 'val')
        dirs={'train':train_dir,'eval':eval_dir, 'model':model_dir, 'pred':''}

        net1 = LayerUNET(device=device,
                             model_depth=4,
                             dataset_zshift=(-50, 200),
                             dirs=dirs,
                             filename=model_name,
                             optimizer='Adam',
                             initial_lr=1e-4,
                             scheduler_patience=3,
                             lossfn=custom_loss_1_smooth,
                             lossfn_smoothness = 50,
                             epochs=40,
                             dropout=True,
                             class_weight=(0.3, 0.7),
                             )

        net1.printConfiguration()
        net1.printConfiguration('logfile')
        print(net1.model)
        print(net1.model, file=net1.logfile)

        net1.train_all_epochs()
        net1.save()


elif mode == 'predict':

    os.environ["CUDA_VISIBLE_DEVICES"]='4'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
   
    pred_dir = '/home/gerlstefan/data/pipeline/selection1/t_rt_mp_gn/tmp/layerseg_prep'
    model_dir ='/home/gerlstefan/models/layerseg/test/mod_191101_depth5.pt'
    out_dir ='/home/gerlstefan/data/layerunet/prediction/191123_depth5_selection1'
    net1 = LayerNetBase(
            dirs={'model': model_dir,
                  'pred': pred_dir,
                  'out': out_dir},
            device=device,
            model_depth=5)

    net1.predict()





