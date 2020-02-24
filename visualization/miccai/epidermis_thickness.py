
import torch
import numpy as np
import os
import nibabel as nib


def load_seg(path):
    img = nib.load(path)
    return img.get_data()

def layer_thickness(pred_dir, files):
    files = [os.path.join(pred_dir, el) for el in files]
    all_values = []
    for f in files:
        A = load_seg(f)
        surf = np.sum(A, axis=0)
        surf = surf.reshape(-1)
        # surf = surf[surf>0]
        all_values = np.concatenate((all_values,surf))
    print(all_values.shape)
    print(os.path.basename(pred_dir), "epidermis thickness")
    print("mean", np.mean(all_values), "std", np.std(all_values))



dirs = ['/home/gerlstefan/data/layerunet/miccai/200201-04-BCE',
        '/home/gerlstefan/data/layerunet/miccai/200202-02-BCE_S_1',
        '/home/gerlstefan/data/layerunet/miccai/200202-03-BCE_S_10',
        '/home/gerlstefan/data/layerunet/miccai/200202-04-BCE_S_100',
        '/home/gerlstefan/data/layerunet/miccai/200202-05-BCE_S_1000',
        '/home/gerlstefan/data/layerunet/miccai/200203-02-BCE_S_2000',
        '/home/gerlstefan/data/layerunet/miccai/200204-00-BCE_S_2500',
        '/home/gerlstefan/data/layerunet/miccai/200204-01-BCE_S_2800']
for pred_dir in dirs:
    files = os.listdir(pred_dir)

    files = [el for el in files if '_pred' in el]

    layer_thickness(pred_dir, files)
