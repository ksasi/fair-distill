from kornia.losses import ssim_loss, psnr_loss
#import lpips
import pywt
import cv2
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset, DataLoader, Dataset
from pytorch_wavelets import DWTForward, DWTInverse
from torch import nn
from pytorch_msssim import ssim, ms_ssim
from focal_frequency_loss import FocalFrequencyLoss as FFL
from torch.distributions import Categorical



class PixelL1Loss(nn.Module):
    """PixelWise L1 Loss as used in https://openaccess.thecvf.com/content/ACCV2020/papers/Chang_TinyGAN_Distilling_BigGAN_for_Conditional_Image_Generation_ACCV_2020_paper.pdf """
    def __init__(self):
        super(PixelL1Loss, self).__init__()
        self.l1loss = nn.L1Loss()

    def forward(self, labels, pred_masks):
        loss = self.l1loss(labels, pred_masks)
        return loss



# Ref : https://github.com/terarachang/ACCV_TinyGAN/blob/837191ff3de9e0c00f43d80e4a46e041af0b6dcd/model.py
class CMPDisLoss(nn.Module):
    def __init__(self):
        super(CMPDisLoss, self).__init__()
        self.criterion = nn.L1Loss() # nn.MSELoss()
        self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]
        
    def forward(self, real_list, fake_list):
        loss = 0
        j = 0
        for i in range(1, len(real_list), 2): # compare actvation values of each layer
            loss += self.weights[j] * self.criterion(fake_list[i], real_list[i])
            j += 1

        return loss


class CSDLoss(nn.Module):
    """
    Cumulative Shannon Diversity Loss for Age, Gender and Race attributes
    computed from pretrained fairface classifier
    """
    def __init__(self):
        super(CSDLoss, self).__init__()
        self.model_fair_7 = torchvision.models.resnet34(pretrained=True)
        self.model_fair_7.fc = nn.Linear(self.model_fair_7.fc.in_features, 18)
        self.model_fair_7.load_state_dict(torch.load('/scratch/data/kotti1/FairFace/fair_face_models/res34_fair_align_multi_7_20190809.pt', map_location=torch.device('cpu')))
        self.model_fair_7 = self.model_fair_7.cuda()
        self.model_fair_7.eval()

        self.trans = transforms.Compose([
        transforms.ToPILImage(),
        transforms.CenterCrop(160),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    class CustomDataset(Dataset):
        def __init__(self, dset, transform=None):
            self.dset = dset
            self.transform = transform

        def __getitem__(self,index):
            image_item = self.dset[index]
            image_item = self.transform(image_item)
            return image_item
        
        def __len__(self):
            return len(self.dset)

    def sd_loss(self, logits):
        """
        Shannon Diversity Loss from logits
        """
        softmax = torch.nn.Softmax(dim=1)
        probs = softmax(logits)
        agmax = torch.argmax(probs, dim = 1)
        binmax = torch.zeros_like(probs).scatter_(1, agmax.unsqueeze(1), 1.)
        freqs = torch.sum(binmax, dim=0)/binmax.shape[0]
        sdi = Categorical(probs = freqs).entropy() # Shannon Diversity Index
        md = torch.log(torch.tensor(freqs.shape[0])) # Maximum Diversity
        sei = sdi/md # Shannon Equitability index
        loss = 1 - sei # Shannon Diversity Loss
        return loss


    def forward(self, images):
        images_dataset = self.CustomDataset(images, self.trans)
        images_loader = torch.utils.data.DataLoader(images_dataset, batch_size=128)
        #images_list = [self.trans(x_).numpy() for x_ in images]
        #images_t = torch.FloatTensor(images_list)
        images_t = next(iter(images_loader))
        logits = self.model_fair_7(images_t.cuda())
        race_logits = logits[:, 0:7]
        gender_logits = logits[:, 7:9]
        age_logits = logits[:, 9:18]
        race_loss = self.sd_loss(race_logits)
        gender_loss = self.sd_loss(gender_logits)
        age_loss = self.sd_loss(age_logits)
        overall_loss = race_loss + gender_loss + age_loss
        return overall_loss




