import torch
import sys
import os
import argparse
import pickle
import PIL.Image

torch.backends.cudnn.benchmark = True

try:
    sys.path.index('/workspace/stylegan2-ada-pytorch')
except:
    sys.path.append('/workspace/stylegan2-ada-pytorch')


with open('/workspace/fair-distill/src/ffhq.pkl', 'rb') as f:
    G = pickle.load(f)['G_ema'].cuda()

print("Size of Latent Dimension z : ", G.z_dim)
