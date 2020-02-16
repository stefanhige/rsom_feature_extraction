
import torch
import os
from laynet._metrics import custom_loss_1_smooth, bce_and_smooth
from laynet import LayerNet, LayerNetBase

# torch.backends.cudnn.benchmark = True
mode = 'predict'

if mode == 'train':
    N = 5

    sdesc = ['FCN_BCE', 'FCN_BCE_S1', 'FCN_BCE_S10', 'FCN_BCE_S100', 'FCN_BCE_S1000']
    # sdesc = ['BCE_S_1', 'BCE_S_10', 'BCE_S_100', 'BCE_S_1000']
    s = [0, 1, 10, 100, 1000]
    root_dir = '/home/gerlstefan/data/layerunet/fullDataset/miccai/crossval/0'

    DEBUG = False
    # DEBUG = True

    out_dir = '/home/gerlstefan/data/layerunet/miccai/fcn/7layer'
    # pred_dir = '/home/gerlstefan/data/pipeline/selection1/t_rt_mp_gn/tmp/layerseg_prep'
    model_type = 'fcn'        
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
                        model_type=model_type,
                        dataset_zshift=(-50, 200),
                        dirs=dirs,
                        optimizer='Adam',
                        initial_lr=1e-4,
                        scheduler_patience=3,
                        lossfn=bce_and_smooth,
                        lossfn_smoothness=s[idx],
                        lossfn_window=5,
                        lossfn_spatial_weight_scale=False,
                        epochs=80,
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

    os.environ["CUDA_VISIBLE_DEVICES"]='5'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
   
    pred_dir = '/home/gerlstefan/data/vesnet/miccai/input_for_layerseg'
    # model_dir ='/home/gerlstefan/models/layerseg/test/mod_191101_depth5.pt'
    model_dir ='/home/gerlstefan/data/layerunet/miccai/fcn/7layer/200210-01-FCN_BCE_S10/mod200210-01.pt'
    out_dir ='/home/gerlstefan/data/vesnet/miccai/layerseg_prediction/fcn/200210-01-FCN_BCE_S10'
    model_type = 'fcn'
    
    net1 = LayerNetBase(
            dirs={'model': model_dir,
                  'pred': pred_dir,
                  'out': out_dir},
            device=device,
            model_depth=5,
            model_type=model_type)
    net1.predict_calc()





