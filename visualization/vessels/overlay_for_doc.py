
import os
import sys
import imageio

if __name__ == '__main__':
    sys.path.append('../../')
from pipeline import vessel_pipeline
from vesnet.deep_vessel_3d import DeepVesselNet


dirs = {'input': '~/data/pipeline/new_data/mat',
        'laynet_model': '~/models/layerseg/test/mod_190731_depth4.pt',
        'vesnet_model': '~/data/vesnet/out/archive2/191003-03-rt_nrsomf_bce_gn/mod191003-03.pt',
        'output': os.getcwd(),
        'figures': '../../doc/figures'}

dirs = {k: os.path.expanduser(v) for k, v in dirs.items()}



# use a standard vesnet with group norm,
# trained on synth+annot

os.environ["CUDA_VISIBLE_DEVICES"]='5'
img = vessel_pipeline(dirs=dirs,
                laynet_depth=4,
                vesnet_model=DeepVesselNet(groupnorm=True),
                ves_probability=0.931, # determined with predict_adj
                divs=(1,1,2),
                pattern=['R_20181124171923_'],  #if list, use patterns, otherwise, use whole dir
                delete_tmp=True,
                return_img=True,
                mip_overlay_axis=1)

imageio.imwrite(os.path.join(dirs['figures'],'mip_refl_noise_no_overlay.png'), img[0])
imageio.imwrite(os.path.join(dirs['figures'],'mip_refl_noise_simple_model.png'), img[1])

# use a standard vesnet with group norm,
# trained on synth+annot+background

dirs['vesnet_model'] = '~/data/vesnet/out/191017-00-rt_+backg_bce_gn/mod191017-00.pt'


dirs = {k: os.path.expanduser(v) for k, v in dirs.items()}

img = vessel_pipeline(dirs=dirs,
                laynet_depth=4,
                vesnet_model=DeepVesselNet(groupnorm=True),
                ves_probability=0.965, # value was checked
                divs=(1,1,2),
                pattern=['R_20181124171923_'],  #if list, use patterns, otherwise, use whole dir
                delete_tmp=True,
                return_img=True,
                mip_overlay_axis=1)

imageio.imwrite(os.path.join(dirs['figures'],'mip_refl_noise_backgr_model.png'), img[1])

# use a more parameter vesnet with group norm
# trained on synth+annot+background


dirs['vesnet_model'] = '~/data/vesnet/out/191108-01-t+rt_mp_gn/mod191108-01.pt'

model = DeepVesselNet(in_channels=2,
                      channels = [2, 10, 20, 40, 80, 1],
                      kernels = [3, 5, 5, 3, 1],
                      depth = 5, 
                      dropout=False,
                      groupnorm=True)

dirs = {k: os.path.expanduser(v) for k, v in dirs.items()}

img = vessel_pipeline(dirs=dirs,
                laynet_depth=4,
                vesnet_model=model,
                ves_probability=0.944, # value was checked
                divs=(1,1,2),
                pattern=['R_20181124171923_'],  #if list, use patterns, otherwise, use whole dir
                delete_tmp=True,
                return_img=True,
                mip_overlay_axis=1)

imageio.imwrite(os.path.join(dirs['figures'],'mip_refl_noise_backgr_mp_model.png'), img[1])
