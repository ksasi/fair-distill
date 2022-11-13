
import os
import random
import argparse
import copy

import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim
from torch import nn


from model import Generator, Discriminator
from loss import PixelL1Loss, CompositeLoss, CMPDisLoss, CSDLoss
from utils import DISTILLEDDataset

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import numpy as np

#from kornia.metrics import ssim, psnr





parser = argparse.ArgumentParser(description='Pytorch framework for GAN Distillation')

parser.add_argument("--dataset_path", default="", type=str, help='path to the dataset(default is None)')
parser.add_argument("--epochs", default=30, type=int, help='epochs for training (default value is 30)')
parser.add_argument("--batch_size", default=128, type=int, help='mini-batch size for training (default value is 128)')
parser.add_argument("--learning_rate", default=1e-2, type=float, help='initial learning rate for training (default value is 0.01)')
parser.add_argument("--momentum", default=0.9, type=float, help='momentum (default value is 0.9)')
parser.add_argument("--weight_decay", default=1e-4, type=float, help='weight decay (default value is 1e-4)')
parser.add_argument("--checkpoint_name", default="", type=str, help='Name of the checkpoint')
parser.add_argument("--save_path", default="", type=str, help='path to save the checkpoint file(default is None)')


#train_epoch(G_Student, D_Student, trainloader_real, criterion, optimizer_g, optimizer_d, lr_scheduler, phase='train')

#def train_epoch(model, trainloader, criterion, optimizer, lr_scheduler, phase='train'):
def train_epoch(G_Student, D_Student, trainloader_real, criterion, optimizer_g, optimizer_d, epoch, phase='train'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    G_Student.train()
    D_Student.train()
    running_loss = 0.0
    epoch_loss_d = 0.
    epoch_loss_g = 0.
    D_x = 0.
    D_G_z1 = 0.
    D_G_z2 = 0.
    #psnr_score_running = 0.
    #ssim_score_running = 0.
    
    batch_num = 0.
    samples_num = 0.
    real_label = 1.
    fake_label = 0.
    alpha = 0.1


    x = np.linspace(0, args.epochs-1, int(args.epochs))
    y = np.linspace(0, 1, int(args.epochs))
    coef = np.polyfit(x,y,1)
    poly1d_fn = np.poly1d(coef)


    for batch_idx, data in enumerate(trainloader_real, 0):
        latents, images_real = data
        #images_real, cropped_images_real = cropped_images_real, images_real
        latents = latents.to(device)
        images_real = images_real.to(device)
        G_Student = G_Student.to(device)
        D_Student = D_Student.to(device)
        #D_Student.zero_grad()

        ### Discriminator - Real Images
        labels = torch.full((images_real.size(0),), real_label-alpha, dtype=torch.float, device=device)
        #print(labels)
        #print(labels.unsqueeze(1))
        optimizer_d.zero_grad()
        preds_real, real_list = D_Student(images_real)
        real_list = [h.detach() for h in real_list]

        #print(preds_real)
        loss_real = criterion[1](preds_real, labels.unsqueeze(1))
        #print(loss_real)
        loss_real.backward()
        D_x  += torch.mean(torch.round(torch.sigmoid(preds_real)).detach().cpu()).item()
        #optimizer_d.step()
        #epoch_loss_d += torch.mean(loss_real.detach().cpu()).item()

        ### Discriminator - Fake Images
        labels.fill_(fake_label)
        images_fake = G_Student(latents)
        #optimizer_d.zero_grad()
        preds_fake, _ = D_Student(images_fake.detach())
        loss_fake = criterion[1](preds_fake, labels.unsqueeze(1))
        loss_fake.backward()
        D_G_z1  += torch.mean(torch.round(torch.sigmoid(preds_fake)).detach().cpu()).item()
        loss_rf = loss_real + loss_fake
        optimizer_d.step()
        #epoch_loss_d += torch.mean(loss_fake.detach().cpu()).item()
        epoch_loss_d += torch.mean(loss_rf.detach().cpu()).item()
        
        if (epoch+1)%1 == 0 :
            images_fake = G_Student(latents)
            ### Generator
            #G_Student.zero_grad()
            labels.fill_(real_label)
            optimizer_g.zero_grad()
            #optimizer_d.zero_grad()
            preds_fake, fake_list = D_Student(images_fake)
            loss1_g = criterion[0](images_real, images_fake)    ### Composite Loss i.e. pixell1_loss
            loss2_g = criterion[1](preds_fake, labels.unsqueeze(1)) ### GAN Loss
            loss3_g = criterion[2](real_list, fake_list) ### Feature-level Distillation Loss
            loss4_g = criterion[3](images_fake) ### Cumulative Shannon Diversity Loss for Age, Gender and Race attributes
            '''
            k = poly1d_fn(epoch)
            if epoch < 249:
            #if epoch < 0:
                loss_g = loss1_g
            else:
                loss_g = (1-k) * loss1_g + loss2_g
            ####loss_g = loss2_g
            #loss_g = 0.4 * loss1_g 
            #optimizer_g.zero_grad()
            '''
            #loss_g = 1.0 * loss1_g + 0.001 * loss2_g
            k = poly1d_fn(epoch)
            loss_g = (1-k) * loss1_g + 0.4 * loss2_g + loss3_g + k * loss4_g
            loss_g.backward()
            D_G_z2 += torch.mean(torch.round(torch.sigmoid(preds_fake)).detach().cpu()).item()
            optimizer_g.step()
            epoch_loss_g += torch.mean(loss_g.detach().cpu()).item()
        batch_num += 1
        samples_num += len(images_fake)
    #return epoch_loss / batch_num, ssim_score_running / samples_num, psnr_score_running/ samples_num
        #break
    return epoch_loss_d / 2*batch_num, epoch_loss_g / batch_num, D_x / batch_num, D_G_z1 / batch_num, D_G_z2 / batch_num

def test_epoch(G_Student, D_Student, testloader_real, criterion, optimizer_g, optimizer_d, phase='test'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    G_Student.eval()

    epoch_loss_g = 0.
    #epoch_acc = 0.
    
    batch_num = 0.
    samples_num = 0.
    #psnr_score_running = 0.
    #ssim_score_running = 0.
    
    #true_labels = []
    #pred_labels = []

    with torch.no_grad():
      for batch_idx, data in  enumerate(testloader_real):

        latents, images_real = data
        #images_real, cropped_images_real = cropped_images_real, images_real
        latents = latents.to(device)
        images_real = images_real.to(device)
        G_Student = G_Student.to(device)
        images_fake = G_Student(latents)


        '''
        loss_fn_alex = lpips.LPIPS(net='alex', verbose = False).cuda()  ### AlexNet perpetual loss
        loss_dist = loss_fn_alex(labels, pred_masks)
        ssim_loss_batch = ssim_loss(labels, pred_masks, 11) ## Include in composite loss
        psnr_loss_batch = psnr_loss(labels, pred_masks, 1) ## Include in composite loss
        composite_loss = 0.6 * loss_dist + 0.1 * ssim_loss_batch + 0.3 * psnr_loss_batch
        #composite_loss = loss_dist
        #loss = torch.mean(loss_dist)
        '''
        #composite_loss = criterion(labels, pred_masks)
        #loss = torch.mean(composite_loss)
        loss = criterion[0](images_real, images_fake)

        ####print(f'\r{phase} batch [{batch_idx}/{len(testloader)}]: loss {torch.mean(loss).item()}', end='', flush=True)
        epoch_loss_g += torch.mean(loss.detach().cpu()).item()
        
        batch_num += 1
        samples_num += len(images_real)
      #return epoch_loss / batch_num, ssim_score_running / samples_num, psnr_score_running/ samples_num
      return epoch_loss_g / batch_num



#def train_model(model, train_loader, test_loader, criterion, optimizer, lr_scheduler, save_path, checkpoint_name, epochs):
def train_model(G_Student, D_Student, trainloader_real, testloader_real, criterion, optimizer_g, optimizer_d, lr_scheduler_g, lr_scheduler_d, save_path, checkpoint_name, epochs):
    train_g_losses = []
    train_d_losses = []
    test_g_losses = []
    #train_ssims = []
    #train_psnrs = []
    #test_losses_id = []
    #test_ssims_id = []
    #test_psnrs_id = []
    #test_losses_od = []
    #test_ssims_od = []
    #test_psnrs_od = []
    best_loss = 0
    best_model = None

    for epoch in range(epochs):
        print('='*15, f'Epoch: {epoch}', flush = True)
    
        #lr_scheduler.step()
        #train_loss, train_ssim, train_psnr = train_epoch(model, train_loader, criterion, optimizer, lr_scheduler, phase='train')

        #test_loss_id, test_ssim_id, test_psnr_id = test_epoch(model, test_loader_id, criterion, optimizer, lr_scheduler, phase='test')

        #test_loss_od, test_ssim_od, test_psnr_od = test_epoch(model, test_loader_od, criterion, optimizer, lr_scheduler, phase='test') 

        #testloader_id
        train_loss_d, train_loss_g, D_x, D_G_z1, D_G_z2 = train_epoch(G_Student, D_Student, trainloader_real, criterion, optimizer_g, optimizer_d, epoch, phase='train')
        #train_loss = train_epoch(model, train_loader, criterion, optimizer, lr_scheduler, phase='train')
        #test_loss = test_epoch(model, test_loader, criterion, optimizer, lr_scheduler, phase='test')
        ########test_loss_g = test_epoch(G_Student, D_Student, testloader_real, criterion, optimizer_g, optimizer_d, phase='test') ### Test Epoch being disabled as this is GAN training

        lr_scheduler_d.step()
        lr_scheduler_g.step()
        print()
        #print(f'Train loss_D: {train_loss_d}', flush = True)
        #print(f'Train loss_G: {train_loss_g}', flush = True)
        print('\tTrain loss_D: %.4f\tTrain loss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f' % (train_loss_d, train_loss_g, D_x, D_G_z1, D_G_z2))
        print()
        ########print(f'Test loss_G: {test_loss_g}', flush = True)
        #print(f'Train loss: {train_loss}, Train SSIM: {train_ssim}, Train PSNR: {train_psnr}', flush = True)
        #print(f'Test loss SOTS Indoor: {test_loss_id}, Test SSIM SOTS Indoor: {test_ssim_id}, Test PSNR SOTS Indoor: {test_psnr_id}', flush = True)
        #print(f'Test loss SOTS Outdoor: {test_loss_od}, Test SSIM SOTS Outdoor: {test_ssim_od}, Test PSNR SOTS Outdoor: {test_psnr_od}', flush = True)
        
    
        train_g_losses.append(train_loss_g)
        train_d_losses.append(train_loss_d)
        ########test_g_losses.append(test_loss_g)
        #train_ssims.append(train_ssim)
        #train_psnrs.append(train_psnr)
        #test_losses_id.append(test_loss_id)
        #test_ssims_id.append(test_ssim_id)
        #test_psnrs_id.append(test_psnr_id)
        #test_losses_od.append(test_loss_od)
        #test_ssims_od.append(test_ssim_od)
        #test_psnrs_od.append(test_psnr_od)
    
        #torch.save({'epoch': epoch, 'model': model.state_dict()}, f'/content/drive/MyDrive/CV_Project/CheckPoints/r3/unet-{epoch}.pt')

        #checkpoint_path = save_path + checkpoint_name + '-' + str(epoch) + '.pt'
        #torch.save({'epoch': epoch, 'model': model.state_dict()}, f'{checkpoint_path}')
        #checkpoint_path = save_path + checkpoint_name + '.pt'
        checkpoint_path_g = save_path + checkpoint_name + '_G-' + str(epoch) + '.pt'
        checkpoint_path_d = save_path + checkpoint_name + '_D-' + str(epoch) + '.pt'
        ####if best_model is None or test_loss < best_loss:
        '''
        if best_model is None or ((train_loss_d < 0.55 and train_loss_d > 0.45) and (train_loss_d < best_loss)):
            best_model = copy.deepcopy(G_Student)
            best_loss = train_loss_d
            ###best_loss = train_loss
            #best_test_acc = test_dice
            #best_pred_labels = pred_labels
            #torch.save({'epoch': epoch, 'model': model.state_dict()}, f'nn-{seed}.pt')
            torch.save({'epoch': epoch, 'model': best_model.state_dict()}, f'{checkpoint_path}')
        '''
        best_model_g = copy.deepcopy(G_Student)
        #best_model_d = copy.deepcopy(D_Student)
        best_model_d = D_Student
        torch.save({'epoch': epoch, 'model': best_model_g.state_dict()}, f'{checkpoint_path_g}')
        torch.save({'epoch': epoch, 'model': best_model_d.state_dict()}, f'{checkpoint_path_d}')
    return train_g_losses, train_d_losses
    ########return train_g_losses, train_d_losses, test_g_losses


def main():
    global args
    args = parser.parse_args()

    torch.backends.cudnn.benchmark = True

    #model_unet = InvolutionUNet(enc_chs=(3,64,128,256,512), dec_chs=(512, 256, 128, 64), num_class=3, retain_dim=False, out_sz=(128,128))
    #####G_Student = Generator(256, 32, 512, 128, 5)
    G_Student = Generator(128, 64, 512, 128, 5)
    D_Student = Discriminator(128, 64, 5)


    train_transform = transforms.Compose([transforms.ToPILImage(), transforms.Resize(128), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    test_transform = transforms.Compose([transforms.ToPILImage(), transforms.Resize(128), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    ####train_transform = transforms.Compose([transforms.ToTensor()])
    ####test_transform = transforms.Compose([transforms.ToTensor()])

    #trainset = RESIDEDataset(path = args.dataset_path + 'RESIDE/', train = True, transform = train_transform, transform_input=transform_input, transform_target=transform_target)
    #testset_id = RESIDEDataset(path = args.dataset_path + 'SOTS/indoor/', train = False, transform = test_transform, transform_input=transform_input, transform_target=transform_target)
    #testset_od = RESIDEDataset(path = args.dataset_path + 'SOTS/outdoor/', train = False, transform = test_transform, transform_input=transform_input, transform_target=transform_target)

    trainset = DISTILLEDDataset(path = args.dataset_path + 'disttrain/', transform = train_transform)
    testset = DISTILLEDDataset(path = args.dataset_path + 'disttest/', transform = test_transform)

    trainloader_real = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=False, num_workers=14, pin_memory=True)
    testloader_real = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=14, pin_memory=True)
    #testloader_od = torch.utils.data.DataLoader(testset_od, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)

    #criterion = CompositeLoss()
    #criterion = PixelL1Loss()
    #criterion1 = CompositeLoss()
    #criterion1 = PixelL1Loss()
    #criterion2 = nn.BCEWithLogitsLoss()
    criterion = [CompositeLoss(), nn.BCEWithLogitsLoss(), CMPDisLoss(), CSDLoss()]
    #criterion = [PixelL1Loss(), nn.BCEWithLogitsLoss()]
    
    #criterion = nn.MSELoss()
    #optimizer = torch.optim.SGD(model_unet.parameters(), lr=1e-2, weight_decay=0.01, momentum=0.9)
    ##optimizer = torch.optim.SGD(model_unet.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay, momentum=args.momentum)
    ##lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)
    optimizer_g = torch.optim.Adam(G_Student.parameters(), lr=args.learning_rate, betas=(0.0, 0.9), weight_decay=args.weight_decay)
    ########optimizer_g = torch.optim.Adam(G_Student.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    optimizer_d = torch.optim.Adam(D_Student.parameters(), lr=args.learning_rate, betas=(0.0, 0.9), weight_decay=args.weight_decay)
    lr_scheduler_g = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer_g,  args.epochs, eta_min=0, verbose=True)
    lr_scheduler_d = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer_d,  args.epochs, eta_min=0, verbose=True)
    #######lr_scheduler_g = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_g, 100, eta_min=0, verbose=True)
    #######lr_scheduler_d = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_d, 100, eta_min=0, verbose=True)
    #######lr_scheduler_g = torch.optim.lr_scheduler.StepLR(optimizer_g, step_size=1, gamma=0.1, verbose=True)
    #######lr_scheduler_d = torch.optim.lr_scheduler.StepLR(optimizer_d, step_size=1, gamma=0.1, verbose=True)
    

    ########train_g_losses, train_d_losses, test_g_losses = train_model(G_Student, D_Student, trainloader_real, testloader_real, criterion, optimizer_g, optimizer_d, lr_scheduler_g, lr_scheduler_d, args.save_path, args.checkpoint_name, args.epochs)
    train_g_losses, train_d_losses = train_model(G_Student, D_Student, trainloader_real, testloader_real, criterion, optimizer_g, optimizer_d, lr_scheduler_g, lr_scheduler_d, args.save_path, args.checkpoint_name, args.epochs)

if __name__ == "__main__":
    main()
