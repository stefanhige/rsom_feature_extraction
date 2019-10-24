"""
Created on Thu Oct 26 11:06:51 2017

@author: Utku Ozbulak - github.com/utkuozbulak
"""
import sys
from PIL import Image
import numpy as np
import nibabel as nib

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

from misc_functions import get_example_params, save_class_activation_images

# to make classes importable
sys.path.append('../../../')
from vesnet.dataloader import RSOMVesselDataset, DropBlue, ToTensor
from vesnet.deep_vessel_3d import DeepVesselNet

class CamExtractor():
    """
        Extracts cam features from the model
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None

    def save_gradient(self, grad):
        self.gradients = grad

    def forward_pass(self, x):
        """
            Does a full forward pass on convolutions, hooks the function at given layer
        """
        conv_output = None
        for module_pos, module in self.model.layers._modules.items():
            x = module(x)  # Forward
            if int(module_pos) == self.target_layer:
                x.register_hook(self.save_gradient)
                conv_output = x  # Save the convolution output on that layer
        return conv_output, x
    

class GradCam():
    """
        Produces class activation map
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.model.eval()
        # Define extractor
        self.extractor = CamExtractor(self.model, target_layer)

    def generate_cam(self, input_image, target_class=None):
        # Full forward pass
        # conv_output is the output of convolutions at specified layer
        # model_output is the final output of the model (1, 1000)
        conv_output, model_output = self.extractor.forward_pass(input_image)
        
        
        #if target_class is None:
        #    target_class = np.argmax(model_output.data.numpy())
        print(target_class)
        # Target for backprop
        one_hot_output = torch.zeros(tuple(model_output.shape), dtype=torch.float32)
        one_hot_output[0][0][target_class] = 1
        #one_hot_output[0,0,30:33,14,40] = 1
        print(one_hot_output.sum())
                
        # Zero grads
        self.model.layers.zero_grad()
        # Backward pass with specified target
        model_output.backward(gradient=one_hot_output, retain_graph=True)
        # Get hooked gradients
        guided_gradients = self.extractor.gradients.data.numpy()[0, ...]
        
        print('guided_gradients shape:', guided_gradients.shape)
        
        # Relu
        #guided_gradients = np.maximum(guided_gradients, 0)
        
        #img = nib.Nifti1Image(guided_gradients[0,...], np.eye(4))
        #nib.save(img, 'gg0_.nii.gz')
        #img = nib.Nifti1Image(guided_gradients[1,...], np.eye(4))
        #nib.save(img, 'gg1_.nii.gz')
        #img = nib.Nifti1Image(guided_gradients[2,...], np.eye(4))
        #return guided_gradients[2, ...]
        #nib.save(img, 'gg2_.nii.gz')
        #img = nib.Nifti1Image(guided_gradients[3,...], np.eye(4))
        #nib.save(img, 'gg3.nii.gz')
        
        img = nib.Nifti1Image(np.pad(model_output.detach().numpy().squeeze(),7,mode='constant'), np.eye(4))
        nib.save(img, 'output.nii.gz')
        
        # Get convolution outputs
        # these are the "forward activation maps"
        target = conv_output.data.numpy()[0, ...]
        
        img = nib.Nifti1Image(np.pad(target[31, ...],6,mode='constant'), np.eye(4))
        nib.save(img, 'target31.nii.gz')
        img = nib.Nifti1Image(np.pad(target[32, ...],6,mode='constant'), np.eye(4))
        nib.save(img, 'target32.nii.gz')
        # Get weights from gradients
        # average over spatial dimensions
        # first dimension are feature maps
        # TODO: average only over "valid" area
        
        #location activation map
#        lam = guided_gradients * target
#        #relu
#        lam[lam<0] = 0
#        
#        img = nib.Nifti1Image(lam[0, ...], np.eye(4))
#        nib.save(img, 'lam0.nii.gz')
#        img = nib.Nifti1Image(lam[1, ...], np.eye(4))
#        nib.save(img, 'lam1.nii.gz')
#        
#        lam = lam / np.amax(lam)
#        lam = np.mean(lam, axis=0)
#        img = nib.Nifti1Image(lam, np.eye(4))
#        nib.save(img, 'lam_mean.nii.gz')
        
        
        
        
        ofs = 1
        weights = np.mean(guided_gradients[:,
                                           target_class[0]-ofs:target_class[0]+ofs,
                                           target_class[1]-ofs:target_class[1]+ofs,
                                           target_class[2]-ofs:target_class[2]+ofs
                                           ], axis=(1, 2, 3))  # Take averages for each gradient
        print(weights)
        print(np.argmax(weights))
        print(weights[31], weights[32])
        
        # add targets with positive weights
        real_weight = self.model.layers[4].weight.detach().numpy().squeeze()
        
        flag = 0
        for i in range(len(weights)):
            f = real_weight[i]
            if real_weight[i] < 0:
                if not flag:
                    target_sum_p = -f*target[i,...]
                    flag = 1
                else:
                    target_sum_p += -f*target[i,...]
                    
        flag = 0
        for i in range(len(weights)):
            f = real_weight[i]
            if real_weight[i] > 0:
                if not flag:
                    target_sum_n = f*target[i,...]
                    flag = 1
                else:
                    target_sum_n += f*target[i,...]
                    
        img = nib.Nifti1Image(np.pad(target_sum_n*target_sum_p,6,mode='constant'), np.eye(4))
        nib.save(img, 'target_sum.nii.gz')
                    
                    
        
        # Create empty numpy array for cam
#        target = target[:,
#                        target_class[0]-5:target_class[0]+5,
#                        target_class[1]-5:target_class[1]+5,
#                        target_class[2]-5:target_class[2]+5,
#                        ]
        
        cam = np.ones(target.shape[1:], dtype=np.float32)
        # Multiply each weight with its conv output and then, sum
        for i, w in enumerate(weights):
            cam[target_class[0]-ofs:target_class[0]+ofs,
                target_class[1]-ofs:target_class[1]+ofs,
                target_class[2]-ofs:target_class[2]+ofs
                ] += w * target[i,
                                target_class[0]-ofs:target_class[0]+ofs,
                                target_class[1]-ofs:target_class[1]+ofs,
                                target_class[2]-ofs:target_class[2]+ofs
                                ]
        cam = np.maximum(cam, 0)
        cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))  # Normalize between 0-1
        cam = np.uint8(cam * 255)  # Scale between 0-255 to visualize
        
        #cam = np.uint8(Image.fromarray(cam), Image.ANTIALIAS))/255
        # ^ I am extremely unhappy with this line. Originally resizing was done in cv2 which
        # supports resizing numpy matrices with antialiasing, however,
        # when I moved the repository to PIL, this option was out of the window.
        # So, in order to use resizing with ANTIALIAS feature of PIL,
        # I briefly convert matrix to PIL image and then back.
        # If there is a more beautiful way, do not hesitate to send a PR.
        return cam



target_class_ = [(30, 14, 40),
                 (31, 14, 40),
                 (32, 14, 40)]
                     # coordinates
target_class_ = [target_class_[0]]

for i, target_class in enumerate(target_class_):
    # load pretrained model
    model_dir = '/home/stefan/data/vesnet/out/191017-00-rt_+backg_bce_gn/mod191017-00.pt'
    model = DeepVesselNet(groupnorm=True) # default settings with group norm
    model.load_state_dict(torch.load(model_dir))
    model.eval()
    
    # load rsom images
    # use standard dataset, this is the easiest
    data_dir = '/home/stefan/data/vesnet/annotatedDataset/train'
    
    dataset = RSOMVesselDataset(data_dir,
                                divs=(1,1,1), 
                                offset=(6,6,6),
                                transform=transforms.Compose([
                                        DropBlue(),
                                        ToTensor()]))
    img = dataset[0]['data']
    print(dataset[0]['meta']['filename'])
    img = img.unsqueeze(0)
    print(img.shape)
    #img = img[:, :, :99, :100, :101]
    img = img[:, :, 0:100, :100, :]
    
    # Vm is a 4-d numpy array, with the last dim holding RGB
    inp = img.numpy().copy()
    inp = np.concatenate((inp,np.zeros((inp.shape[0], 1, inp.shape[2], inp.shape[3], inp.shape[4]))),axis=1)
    inp = inp.squeeze()
    inp = np.moveaxis(inp, 0, -1)
    inp = np.ascontiguousarray(inp)
    print(inp.shape)
    print(inp.dtype)
    shape_3d = inp.shape[0:3]
    rgb_dtype = np.dtype([('R', 'u1'), ('G', 'u1'), ('B', 'u1')])
    inp = inp.astype(np.uint8)
    inp = inp.view(rgb_dtype)
    inp = inp.reshape(shape_3d)
    inp = nib.Nifti1Image(inp, np.eye(4))
    nib.save(inp, 'input.nii.gz')
    
        
        
    # Grad cam
    grad_cam = GradCam(model, target_layer=3)
    # Generate cam mask
    #cam = grad_cam.generate_cam(img, target_class)
    
    if i == 0:
        gg = grad_cam.generate_cam(img, target_class)
    else:
        gg += grad_cam.generate_cam(img, target_class)
    
    
    # Save mask
    #save_class_activation_images(original_image, cam, file_name_to_export)
    
#    img = nib.Nifti1Image(cam, np.eye(4))
#    nib.save(img, 'grad_cam_test.nii.gz')
    
    print('Grad cam completed')


img = nib.Nifti1Image(gg, np.eye(4))
nib.save(img, 'lin_test.nii.gz')


























