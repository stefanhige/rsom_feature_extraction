"""
Created on Thu Oct 26 11:06:51 2017

@author: Utku Ozbulak - github.com/utkuozbulak
"""
import sys
from PIL import Image
import numpy as np
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
        
        
        if target_class is None:
            target_class = np.argmax(model_output.data.numpy())
        # Target for backprop
        one_hot_output = torch.zeros(tuple(model_output.shape), dtype=torch.float32)
        one_hot_output[0][0][target_class] = 1
        # Zero grads
        self.model.layers.zero_grad()
        # Backward pass with specified target
        model_output.backward(gradient=one_hot_output, retain_graph=True)
        # Get hooked gradients
        guided_gradients = self.extractor.gradients.data.numpy()[0,0, ...]
        # Get convolution outputs
        # TODO:
        # works till here,
        # what is target used for??
        target = conv_output.data.numpy()[0,0, ...]
        # Get weights from gradients
        weights = np.mean(guided_gradients, axis=(1, 2))  # Take averages for each gradient
        # Create empty numpy array for cam
        cam = np.ones(target.shape[1:], dtype=np.float32)
        # Multiply each weight with its conv output and then, sum
        for i, w in enumerate(weights):
            cam += w * target[i, :, :]
        cam = np.maximum(cam, 0)
        cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))  # Normalize between 0-1
        cam = np.uint8(cam * 255)  # Scale between 0-255 to visualize
        cam = np.uint8(Image.fromarray(cam).resize((input_image.shape[2],
                       input_image.shape[3]), Image.ANTIALIAS))/255
        # ^ I am extremely unhappy with this line. Originally resizing was done in cv2 which
        # supports resizing numpy matrices with antialiasing, however,
        # when I moved the repository to PIL, this option was out of the window.
        # So, in order to use resizing with ANTIALIAS feature of PIL,
        # I briefly convert matrix to PIL image and then back.
        # If there is a more beautiful way, do not hesitate to send a PR.
        return cam



target_class = (50, 50, 50) # coordinates
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
img = img.unsqueeze(0)
img = img[:, :, :100, :100, :100]
    
    
# Grad cam
grad_cam = GradCam(model, target_layer=1)
# Generate cam mask
cam = grad_cam.generate_cam(img, target_class)
# Save mask
save_class_activation_images(original_image, cam, file_name_to_export)
print('Grad cam completed')
