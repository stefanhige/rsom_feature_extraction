
import torch
import os
from laynet._metrics import custom_loss_1_smooth, bce_and_smooth
from laynet import LayerNet, LayerNetBase

# torch.backends.cudnn.benchmark = True
mode = 'train'

if mode == 'train':
    N = 1

    # sdesc = ['s100-w3','s1000-w3','s10000-w3']
    sdesc = ['BCE']

    root_dir = '/home/gerlstefan/data/layerunet/fullDataset/miccai/crossval/0'

    DEBUG = False
    # DEBUG = True

    out_dir = '/home/gerlstefan/data/layerunet/miccai'
    # pred_dir = '/home/gerlstefan/data/pipeline/selection1/t_rt_mp_gn/tmp/layerseg_prep'
            
    os.environ["CUDA_VISIBLE_DEVICES"]='4'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for idx in range(N):
        root_dir = root_dir

        # train_dir = '/home/gerlstefan/data/layerunet/dataloader_dev/1'
        # eval_dir = '/home/gerlstefan/data/layerunet/dataloader_dev/2'
        train_dir = os.path.join(root_dir, 'train')
        eval_dir = os.path.join(root_dir, 'val')
        pred_dir = eval_dir
        dirs={'train':train_dir,'eval':eval_dir, 'model':'', 'pred':pred_dir, 'out': out_dir}

        net1 = LayerNet(device=device,
                        sdesc=sdesc[idx],
                        model_depth=5,
                        dataset_zshift=(-50, 200),
                        dirs=dirs,
                        optimizer='Adam',
                        initial_lr=1e-4,
                        scheduler_patience=3,
                        lossfn=bce_and_smooth,
                        lossfn_smoothness=0,
                        lossfn_window=5,
                        lossfn_spatial_weight_scale=False,
                        epochs=1,
                        dropout=True,
                        class_weight=None,
                        DEBUG=DEBUG,
                        probability=0.5,
                        slice_wise=False
                         )

        net1.printConfiguration()
        net1.printConfiguration('logfile')

        net1.save_code_status()
        net1.train_all_epochs()
        net1.predict_calc()
        net1.save_model()


elif mode == 'predict':

    os.environ["CUDA_VISIBLE_DEVICES"]='6'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
   
    pred_dir = '/home/gerlstefan/data/layerunet/fullDataset/labeled/val'
    # model_dir ='/home/gerlstefan/models/layerseg/test/mod_191101_depth5.pt'
    model_dir ='/home/gerlstefan/data/layerunet/miccai/200126-00-s1000/mod200126-00.pt'
    out_dir ='/home/gerlstefan/data/layerunet/miccai/dataset-cleanup/eval'
    
    net1 = LayerNetBase(
            dirs={'model': model_dir,
                  'pred': pred_dir,
                  'out': out_dir},
            device=device,
            model_depth=5)
    net1.predict()





