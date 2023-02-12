import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from torch.nn.utils import spectral_norm



# Ref : https://github.com/terarachang/ACCV_TinyGAN

class GBlock(nn.Module):
    """Convolution blocks for the generator"""
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=True):
        super(GBlock, self).__init__()
        hidden_channel = out_channel 
        
        # depthwise seperable
        self.dw_conv1 = nn.Conv2d(in_channel, in_channel,
            kernel_size=kernel_size, stride=stride, padding=padding, bias=bias, groups=int(in_channel))

        self.dw_conv2 = nn.Conv2d(hidden_channel, hidden_channel, 
            kernel_size=kernel_size, stride=stride, padding=padding, bias=bias, groups=int(hidden_channel))
        
        self.pw_conv1 = nn.Conv2d(in_channel, hidden_channel, kernel_size=1)
        self.pw_conv2 = nn.Conv2d(hidden_channel, out_channel, kernel_size=1)

        self.c_sc = nn.Conv2d(in_channel, out_channel, kernel_size=1)

        self.cbn0 = nn.BatchNorm2d(in_channel, affine=False)
        self.cbn1 = nn.BatchNorm2d(hidden_channel, affine=False)
        
        self._initialize()
        
    def _initialize(self):
        nn.init.xavier_uniform_(self.dw_conv1.weight, gain=math.sqrt(2))
        nn.init.xavier_uniform_(self.dw_conv2.weight, gain=math.sqrt(2))
        nn.init.xavier_uniform_(self.pw_conv1.weight, gain=1)
        nn.init.xavier_uniform_(self.pw_conv2.weight, gain=1)
        nn.init.xavier_uniform_(self.c_sc.weight, gain=1)


    def _upsample(self, x):
        h, w = x.size()[2:]
        return F.interpolate(x, size=(h * 2, w * 2), mode='bilinear')

    def shortcut(self, x):
        h = self._upsample(x)
        h = self.c_sc(h)
        return h

    def forward(self, x):
        out = self.cbn0(x)
        out = F.relu(out)
        
        out = self._upsample(out)
        out = self.pw_conv1(self.dw_conv1(out))
        out = self.cbn1(out)
        out = F.relu(out)
        out = self.pw_conv2(self.dw_conv2(out))
        return out + self.shortcut(x)  # residual


class Generator(nn.Module):
    def __init__(self, image_size=128, conv_dim=64, z_dim=128, c_dim=128, repeat_num=5):
        super(Generator, self).__init__()
        self.conv_dim = conv_dim
        self.repeat_num = repeat_num
        self.nfilter0 = np.power(2, repeat_num-1)*self.conv_dim
        self.W0 = image_size // np.power(2, repeat_num)
        
        self.fc = nn.Linear(z_dim, self.nfilter0*self.W0*self.W0)
        nfilter = self.nfilter0
        blocks = []
        blocks.append(GBlock(nfilter, nfilter, kernel_size=3))
        for i in range(1, repeat_num):
            blocks.append(GBlock(nfilter, nfilter//2))
            nfilter = nfilter // 2
        self.blocks = nn.Sequential(*blocks)
        
        self.bn = nn.BatchNorm2d(nfilter)
        self.colorize = nn.Conv2d(conv_dim, 3, kernel_size=3, stride=1, padding=1)

    def forward(self, noise):
        h = self.fc(noise).view(-1, self.nfilter0, self.W0, self.W0)

        for i in range(self.repeat_num):
            h = self.blocks[i](h)
        h = F.relu(self.bn(h))
         
        out = F.tanh(self.colorize(h)) # (batch_size, 3, image_size, image_size)
        
        return out

    def interpolate(self, noise):
        h = self.fc(noise).view(-1, self.nfilter0, self.W0, self.W0)
        
        for i in range(self.repeat_num):
            h = self.blocks[i](h)
        h = F.relu(self.bn(h))
         
        out = F.tanh(self.colorize(h)) # (batch_size, 3, image_size, image_size)
        
        return out



class Discriminator(nn.Module):
    def __init__(self, image_size=128, conv_dim=64, repeat_num=5):
        super(Discriminator, self).__init__()
        y_dim = 2**(repeat_num-1) * conv_dim # default: 1024

        layers = []
        layers.append(spectral_norm(nn.Conv2d(3, conv_dim, kernel_size=4, stride=2, padding=1)))
        layers.append(nn.LeakyReLU(0.01))

        curr_dim = conv_dim
        for i in range(1, repeat_num):
            layers.append(spectral_norm(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1)))
            layers.append(nn.LeakyReLU(0.01))
            curr_dim = curr_dim * 2

        self.layers = nn.Sequential(*layers)

        self.fc_src = spectral_norm(nn.Linear(y_dim, 1))

    def forward(self, x):
        h_list = []
        h = x
        for layer in self.layers:
            h = layer(h)
            h_list.append(h)
        
        h = torch.sum(h, dim=(2,3)) # (bs, 1024) # pooling
        
        out_src = self.fc_src(h)    # (bs, 1)
        return out_src, h_list    # (bs, 1)

