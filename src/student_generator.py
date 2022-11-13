import torch
import sys
import os
import argparse
import pickle
import PIL.Image
from PIL import Image
import numpy as np
from model import Generator
import torchvision.transforms as transforms

# Ref : https://github.com/NVlabs/stylegan2-ada-pytorch

parser = argparse.ArgumentParser(description='Distilled Student generator for generating face images')

parser.add_argument("--num", default=50, type=int, help='Number of images to be generated(default value is 50)')
parser.add_argument("--stimgdir", default="/scratch/data/kotti1/fair-distill/data/stdata/st_images", type=str, help='path to save the generated images from student generator')
parser.add_argument("--stlatdir", default="/scratch/data/kotti1/fair-distill/data/stdata/st_latents", type=str, help='path to save the latents')
parser.add_argument("--teimgdir", default="/scratch/data/kotti1/fair-distill/data/stdata/te_images", type=str, help='path to save the generated images from teacher generator(for same latents)')
parser.add_argument("--model_checkpoint", default="/scratch/data/kotti1/fair-distill/checkpoints/student_GAN.pt", type=str, help='Checkpoint file along with full path')


def main():
    global args
    args = parser.parse_args()

    torch.backends.cudnn.benchmark = True

    
    try:
        sys.path.index('/scratch/data/kotti1/stylegan2-ada-pytorch')
    except:
        sys.path.append('/scratch/data/kotti1/stylegan2-ada-pytorch')
    

    #print(sys.path)
    os.makedirs(args.stimgdir, exist_ok=True)
    os.makedirs(args.stlatdir, exist_ok=True)
    os.makedirs(args.teimgdir, exist_ok=True)


    with open('/scratch/data/kotti1/fair-distill/src/ffhq.pkl', 'rb') as f:
        Teacher_G = pickle.load(f)['G_ema'].cuda()


    ###model_g = Generator(256, 32, 512, 128, 5).cuda()
    model_g = Generator(128, 64, 512, 128, 5).cuda()
    model_g.load_state_dict(torch.load(args.model_checkpoint)['model'])
    target_transform = transforms.Compose([transforms.Normalize((-1.0, -1.0, -1.0), (2.0, 2.0, 2.0))])


    for img_num in range(args.num):
        z = torch.randn([1, Teacher_G.z_dim]).cuda()
        np.save(f'{args.stlatdir}/{img_num}.npy', z.cpu().numpy())
        c = None
        img_teacher = Teacher_G(z, c, truncation_psi=1, noise_mode='const')
        img_teacher = (img_teacher.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        img_teacher = PIL.Image.fromarray(img_teacher[0].cpu().numpy(), 'RGB')
        img_teacher.resize(size=(256, 256)).save(f'{args.teimgdir}/{img_num}_teacher.png')

        #PIL.Image.fromarray(img_teacher[0].cpu().numpy(), 'RGB').save(f'{args.teimgdir}/{img_num}_teacher.png')
        #z = torch.randn([1, 512]).cuda()
        #print(np.load('/workspace/fair-distill/data/alltrain/latentsz/19995.npy'))
        #z = np.load('/workspace/fair-distill/data/alltrain/latentsz/29918.npy')
        #######################z = np.load('/scratch/data/kotti1/fair-distill/data/disttrain/s_latentsz/29918.npy')
        #######################z = torch.from_numpy(z).cuda()
        #print(z)
        img_student = target_transform(model_g(z))
        ####img_student = model_g(z)
        #print(img_student+1)
        #print(((img_student+1).permute(0, 2, 3, 1) * 255).clamp(0, 255).to(torch.uint8))
        #print(img_student.shape)
        img_student = ((img_student).permute(0, 2, 3, 1) * 255).clamp(0, 255).to(torch.uint8)
        img_student = PIL.Image.fromarray(img_student[0].cpu().numpy(), 'RGB')
        img_student.resize(size=(256, 256)).save(f'{args.stimgdir}/{img_num}_student.png')
        #print(img.permute(0, 2, 3, 1) * 255)
        #img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        #img = (img.permute(0, 2, 3, 1) * 255).clamp(0, 255).to(torch.uint8)
        #print(img[0])
        #print("\n")
        #print(img[0].cpu().numpy())
        #####PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB').save(f'{args.imgdir}/{img_num}.png')
        #img = PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB')
        #img.resize(size=(256, 256)).save(f'{args.outdir}/{img_num}.png')
        #np.save(f'{args.zdir}/{img_num}.npy', z.cpu().numpy())

        
if __name__ == "__main__":
    main()

    

