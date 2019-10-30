import torch
import torch.nn.functional as F
import torch.nn as nn
import sys
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from math import floor, ceil
# this whole file is copied from
# https://github.com/ramialmask/pytorch_deepvessel/tree/master/components
# thanks to Rami Al-Maskari



#Define Network
class Deep_Vessel_Net_FC(nn.Module):

    def __init__(self, 
                 in_channels=2, 
                 print_=False): #whats with n_classes? 
        super(Deep_Vessel_Net_FC, self).__init__()
        self.in_channels = in_channels

        # ONLY FOR TESTING PURPOSES
        self.print=print_

        # 1. 3x3x3-Conv, 2 -> 5
        self.conv1 = nn.Conv3d(self.in_channels, 5, kernel_size=(3,3,3)) #nn.Conv3d
        # 2. 5x5x5-Conv, 5 -> 10
        self.conv2 = nn.Conv3d(5, 10, kernel_size=(5,5,5))
        # 3. 5x5x5-Conv, 10-> 20
        self.conv3 = nn.Conv3d(10, 20, kernel_size=(5,5,5))
        # 4. 3x3x3-Conv, 20-> 50
        self.conv4 = nn.Conv3d(20, 50, kernel_size=(3,3,3))
        # 5. FC
        self.conv5 = nn.Conv3d(50,1, kernel_size=(1,1,1))#, bias=False)

    def set_padding(self, size):
        """ONLY FOR TESTING PURPOSES
        """
        self.padding = nn.ReplicationPad3d(size)
        
    def calc_shrink(self):
        X = torch.ones([1, self.in_channels, 25, 25, 25])
        Y = self.forward(X)
        Xshp = torch.tensor(list(X.shape))
        Yshp = torch.tensor(list(Y.shape))
        print(' ',list(Xshp.numpy()),'input dimension')
        print('-',list(Yshp.numpy()),'output dimension')
        print('______________________')
        print('=',list((Xshp-Yshp).numpy()))

    def print_volume(self, x, layer):
        """ONLY FOR TESTING PURPOSES
        set self.c = 0 in init
        set self.c = self.c + 1 in forward
        """
        if self.c == 30 and self.print:
            image = x.data.clone()
            image = image.detach().cpu().numpy()
            print("Image shape before {}".format(image.shape))
            image = image[0,0,:,:,0]
            print("Image shape after {}".format(image.shape))
            image = Image.fromarray(image)
            img_path = "segmentation/output/synth/wave/" + "image_{}_{}.tiff".format(self.c, layer)
            image.save(img_path)
            print("Saved image to {}".format(img_path))

    def forward(self, x):
        
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.conv5(x)
        return x

    def weight_hist(self, gs):
        for i, c in enumerate(self.modules()):
            #print("{} {}".format(i,c._get_name()))
            if i == 0:
                pass
            else:
                # print("plotting {}".format(i))
                w = c.weight.clone().detach().cpu().numpy()
                w = w.ravel()
                me = np.mean(w)
                std = np.std(w)
                mi = np.amin(w)
                ma = np.amax(w)
                # print("{}, {}".format(str(mi),str(ma)))
                le = len(w)
                plt.subplot(gs[i-1,0])
                plt.hist(w, bins=100,facecolor="g",label="mean = {:.4f}\n \
                        std = {:.4f}\n \
                        [{:.4f},{:.4f}]\n \
                        len = {}\n".format(me, std, mi, ma, le))
                plt.legend(loc="best", handletextpad=-2.0, handlelength=0)
                plt.title("{} {} Weight".format(c._get_name(), str(i)))
                if not i == 6:
                    b = c.bias.clone().detach().cpu().numpy()
                    b = b.ravel()
                    me = np.mean(b)
                    std = np.std(b)
                    mi = np.amin(b)
                    ma = np.amax(b)
                    le = len(b)
                    plt.subplot(gs[i-1,1])
                    plt.hist(b, bins=100,facecolor="g",label="mean = {:.4f}\n \
                        std = {:.4f}\n \
                        [{:.4f},{:.4f}]\n \
                        len = {}\n".format(me, std, mi, ma, le))
                    plt.legend(loc="best", handletextpad=-2.0, handlelength=0)
                    plt.title("{} {} Bias".format(c._get_name(), str(i)))

    def print_weights(self, settings):
        """ONLY FOR TESTING PURPOSES
        """
        from utilities.filehandling import writeNifti
        import os
        path="./segmentation/output/test_dir/networks/" + settings["paths"]["input_model"][:-4] + "/"
        os.mkdir(path)
        print("\t\tWeights\t\t\t\tBias\t\tnelements\tsize")
        for i, c in enumerate(self.modules()):
            try:
                writeNifti(path + "layer_{}.nii".format(i), c.weight.clone().detach().cpu().numpy())
                if i == 1:
                    print("{}\t:{}\t{}\t\t{}\t\t{}".format(str(i), \
                            c.weight.shape,
                            c.bias.shape,
                            c.weight.nelement() + c.bias.nelement(),
                            c.weight.nelement() * c.weight.element_size() + \
                            c.bias.nelement() * c.bias.element_size()))
                # if i == 2:
                #     print("First layer weight {}".format(c.weight[-1]))
                if i > 1:
                    print("{}\t:{}\t{}\t{}\t\t{}".format(str(i), \
                            c.weight.shape,
                            c.bias.shape,
                            c.weight.nelement() + c.bias.nelement(),
                            c.weight.nelement() * c.weight.element_size() + \
                            c.bias.nelement() * c.bias.element_size()))
            except AttributeError:
                pass

    def print_layer(self, num):
        module = None
        for i, c in enumerate(self.modules()):
            if i == num:
                module = c
                break
        print("Layer {} Weight {}".format(num, module.weight[-1]))

    def get_module_length(self):
        return len(list(self.modules()))

    def get_layer(self, num):
        module = None
        for i, c in enumerate(self.modules()):
            if i == num:
                module = c
                break
        return module

    def save_model(self, path):
        torch.save(self.state_dict(), path)

    def load_model(self, path):
        t_ = torch.load(path)
        print("T_ {}".format(t_))
        self.load_state_dict(t_)
        print("Loaded model from {0}".format(str(path)))

class DeepVesselNet(nn.Module):

    def __init__(self, 
            in_channels=2,
            channels = [2, 5, 10, 20, 50, 1],
            kernels = [3, 5, 5, 3, 1],
            depth = 5, 
            dropout=False, 
            groupnorm=False):
        super(DeepVesselNet, self).__init__()
       
        print('Calling DeepVesselNet init')
        # SETTINGS
        self.in_channels = in_channels
        self.depth = depth
        self.dropout = dropout
       
        # generate dropout list for every layer
        if dropout:
            self.dropouts = [0, 0, 0.3, 0.3, 0]
        else:
            self.dropouts = [0] * depth

        # generate channels list for every layer
        self.channels = channels
        # override in_channels
        self.channels[0] = in_channels

        # generate kernel size list
        self.kernels = kernels
        
        self.groupnorm = groupnorm
        if groupnorm:
            self.groupnorms = [0] + [1]*(depth-2) + [0]
        else:
            self.groupnorms = [0]*(depth)

        assert len(self.dropouts) == depth
        assert len(self.channels) == depth + 1
        assert len(self.kernels) == depth
        assert len(self.groupnorms) == depth

        layers = []
        # layers = nn.ModuleList()

        # TODO fix notation depth layers?
        
        # deep layers
        for i in range(depth-1):
            layers.append(DVN_Block(
                self.channels[i],
                self.channels[i+1],
                self.kernels[i],
                self.groupnorms[i],
                self.dropouts[i]))
        # last layer
        layers.append(nn.Conv3d(self.channels[-2], self.channels[-1], self.kernels[-1])) 

        self.layers = nn.Sequential(*layers)

    def forward(self, x):

        return self.layers(x)

    def count_parameters(self):

        return sum(p.numel() for p in self.parameters() if p.requires_grad)

class DVN_Block(nn.Module):

    def __init__(self, in_size, out_size, kernel_size, groupnorm, dropout):
        super(DVN_Block, self).__init__()
        
        block = []

        block.append(nn.Conv3d(in_size, out_size, kernel_size))
        block.append(nn.ReLU())
        if groupnorm:
            block.append(nn.GroupNorm(5, out_size))
        if dropout:
            block.append(nn.Dropout3d(dropout))
        
        self.block = nn.Sequential(*block)

    def forward(self, x):
        
        return self.block(x)


class V_Block(nn.Module):

    def __init__(self, in_size, out_size, kernel_size_conv, kernel_size_pool):
        super(V_Block, self).__init__()

        assert kernel_size_conv % 2
        
        self.ksp = kernel_size_pool
        
        straight = []
        straight.append(nn.Conv3d(in_size, floor(out_size/2), kernel_size_conv))
        straight.append(nn.ReLU())
        
        self.straight = nn.Sequential(*straight)
        padding = 0

        down = []
        down.append(nn.MaxPool3d(kernel_size_pool, padding=0))
        down.append(nn.Conv3d(in_size, ceil(out_size/2), kernel_size_conv))
        down.append(nn.ReLU())
        # down.append(nn.ReplicationPad3d(floor(kernel_size_conv/2)))
        
        self.down = nn.Sequential(*down)

        up = []
        up.append(nn.Upsample(scale_factor=kernel_size_pool, mode='trilinear'))
        pad = (kernel_size_pool-1) * floor(kernel_size_conv/2)
        print(pad)
        up.append(nn.ReplicationPad3d(pad))

        self.up = nn.Sequential(*up)

    def forward(self, x):
        
        print('in:',x.shape)
        bridge = self.straight(x)
        print('bridge:', bridge.shape)
        
        # pad to the next multiplier of kernel_size_pool
        pad = (floor(-x.shape[-1] % self.ksp / 2), 
               ceil(-x.shape[-1] % self.ksp / 2), 
               floor(-x.shape[-2] % self.ksp / 2), 
               ceil(-x.shape[-2] % self.ksp / 2), 
               floor(-x.shape[-3] % self.ksp / 2), 
               ceil(-x.shape[-3] % self.ksp / 2))

        x = nn.functional.pad(x, pad)
        print('after pad:', x.shape)

        x = self.down(x)
        print('after down:', x.shape)
        x = self.up(x)
        x = nn.functional.pad(x, tuple(-el for el in pad))
        print('after up:', x.shape)
        return torch.cat((x, bridge), 1)


if __name__ == '__main__':
    
    # test V_Block
    # batch x channels x X x Y x Z
    x = torch.rand((1, 1, 24, 28, 28))

    m = V_Block(1, 2, kernel_size_conv=3, kernel_size_pool=5)

    y = m(x)

        







