import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from torch.nn.utils import spectral_norm



class Block_en(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding = 1)
        self.inonv1 = Involution2d(in_channels=in_ch, out_channels=out_ch, kernel_size = (3,3), padding = (1,1))
        self.relu  = nn.ReLU()
        self.splus = nn.Softplus()
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding = 1)
        self.inonv2 = Involution2d(in_channels=out_ch, out_channels=out_ch, kernel_size = (3,3), padding = (1,1))
    
    def forward(self, x):
        #return self.relu(self.conv2(self.relu(self.conv1(x))))
        #print(self.inonv1(x).shape)
        ######return self.relu(self.conv2(self.relu(self.inonv1(x))))
        return self.splus(self.conv2(self.splus(self.inonv1(x))))
        #return self.relu(self.inonv2(self.relu(self.inonv1(x))))


class Block_de(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding = 1)
        #self.inonv1 = Involution2d(in_channels=in_ch, out_channels=out_ch, kernel_size = (3,3), padding = (1,1))
        self.relu  = nn.ReLU()
        self.splus = nn.Softplus()
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding = 1)
        #self.inonv2 = Involution2d(in_channels=out_ch, out_channels=out_ch, kernel_size = (3,3), padding = (1,1))
    
    def forward(self, x):
        return self.splus(self.conv2(self.splus(self.conv1(x))))
        #print(self.inonv1(x).shape)
        #return self.relu(self.conv2(self.relu(self.inonv1(x))))

class Encoder(nn.Module):
    def __init__(self, chs=(3,64,128,256,512,1024)):
        super().__init__()
        self.enc_blocks = nn.ModuleList([Block_en(chs[i], chs[i+1]) for i in range(len(chs)-1)])
        self.pool       = nn.MaxPool2d(2)
    
    def forward(self, x):
        ftrs = []
        for block in self.enc_blocks:
            x = block(x)
            ftrs.append(x)
            x = self.pool(x)
        return ftrs


class Decoder(nn.Module):
    def __init__(self, chs=(1024, 512, 256, 128, 64)):
        super().__init__()
        self.chs         = chs
        self.upconvs    = nn.ModuleList([nn.ConvTranspose2d(chs[i], chs[i+1], 2, 2) for i in range(len(chs)-1)])
        self.dec_blocks = nn.ModuleList([Block_de(chs[i], chs[i+1]) for i in range(len(chs)-1)]) 
        
    def forward(self, x, encoder_features):
        for i in range(len(self.chs)-1):
            x        = self.upconvs[i](x)
            enc_ftrs = self.crop(encoder_features[i], x)
            x        = torch.cat([x, enc_ftrs], dim=1)
            x        = self.dec_blocks[i](x)
        return x
    
    def crop(self, enc_ftrs, x):
        _, _, H, W = x.shape
        enc_ftrs   = torchvision.transforms.CenterCrop([H, W])(enc_ftrs)
        return enc_ftrs

class InvolutionUNet(nn.Module):
    def __init__(self, enc_chs=(3,64,128,256,512), dec_chs=(512, 256, 128, 64), num_class=1, retain_dim=False, out_sz=(572,572)):
        super().__init__()
        self.encoder     = Encoder(enc_chs)
        self.decoder     = Decoder(dec_chs)
        self.head        = nn.Conv2d(dec_chs[-1], num_class, 1)
        self.retain_dim  = retain_dim
        self.out_sz = out_sz

    def forward(self, x):
        enc_ftrs = self.encoder(x)
        out      = self.decoder(enc_ftrs[::-1][0], enc_ftrs[::-1][1:])
        out      = self.head(out)
        if self.retain_dim:
            out = F.interpolate(out, self.out_sz)
        return out



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
        
        #weight = torch.FloatTensor(np.load('cls_weight_reduce.npy'))
        #self.embeding = nn.Embedding.from_pretrained(weight, freeze=False)
        
        self.fc = nn.Linear(z_dim, self.nfilter0*self.W0*self.W0)
        # after reshape: (N, self.nfilter0, self.W0, self.W0) = (N, 1024, 4, 4)
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
        #y_emb = self.embeding(label)

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

#  G = Generator(config.image_size, config.g_conv_dim, config.z_dim, config.c_dim, config.g_repeat_num)
#  G = Generator(128, 32, 128, 128, 5) and  G = Generator(128, 32, 512, 128, 5)
#  summary(G.cuda(), (128, 128)) and summary(G.cuda(), (128, 512)), Larger size z and image sizes causing CUDA OOM errors in Colab
#  summary(G.cuda(), (128, 512)) and interpolation can give image size of 1024x1024
#  Size of Latent Dimension z :  512


class Discriminator(nn.Module):
    def __init__(self, image_size=128, conv_dim=64, repeat_num=5):
        super(Discriminator, self).__init__()
        y_dim = 2**(repeat_num-1) * conv_dim # default: 1024
        ##self.embeding = spectral_norm(nn.Embedding(1000, y_dim)) 

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
        #h_list = []
        h = x
        for layer in self.layers:
            h = layer(h)
        #    h_list.append(h)
        
        h = torch.sum(h, dim=(2,3)) # (bs, 1024) # pooling
        
        out_src = self.fc_src(h)    # (bs, 1)
        ##out_cls = torch.sum(h * self.embeding(label), dim=1, keepdim=True)
        return out_src    # (bs, 1)
        #return out_src, h_list    # (bs, 1)


# Discriminator(128, 32, 5)
# Discriminator(Image_Size, Conv_dim, Repeat_num)
