from kornia.losses import ssim_loss, psnr_loss
#import lpips
import pywt
import cv2
import torch
from pytorch_wavelets import DWTForward, DWTInverse
from torch import nn
from pytorch_msssim import ssim, ms_ssim
from focal_frequency_loss import FocalFrequencyLoss as FFL

class DWTL2Loss(nn.Module):
    """DWT L2 loss function"""    
    def __init__(self):
        super(DWTL2Loss, self).__init__()
        self.loss = nn.MSELoss().cuda()
        self.dwt = DWTForward(J=3, wave='db1', mode='zero').cuda()

    def forward(self, labels, pred_masks):
        (yl_labels, yh_labels) = self.dwt(labels)
        (yl_pred_masks, yh_pred_masks) = self.dwt(pred_masks)
        dwtl2_loss = self.loss(yh_labels[0], yh_pred_masks[0]) + self.loss(yh_labels[1], yh_pred_masks[1]) + self.loss(yh_labels[2], yh_pred_masks[2])
        return dwtl2_loss


class CompositeOLoss(nn.Module):
    """Composite loss function"""    
    def __init__(self):
        super(CompositeLoss, self).__init__()
        #self.loss_fn_alex = lpips.LPIPS(net='alex', verbose = False).cuda()
        self.loss_fn_vgg = lpips.LPIPS(net='vgg', verbose = False).cuda()
        self.ssim_loss = ssim_loss
        self.psnr_loss = psnr_loss
        self.dwtl2_loss = DWTL2Loss()

    def forward(self, labels, pred_masks):
        #composite_loss = 0.4 * self.loss_fn_alex(labels, pred_masks) + 0.1 * self.ssim_loss(labels, pred_masks, 11) + 0.2 * self.psnr_loss(labels, pred_masks, 1) + 0.3 * self.dwtl2_loss(labels, pred_masks)
        composite_loss = 0.3 * self.loss_fn_vgg(labels, pred_masks) + 0.2 * self.ssim_loss(labels, pred_masks, 11) + 0.2 * self.psnr_loss(labels, pred_masks, 1) + 0.3 * self.dwtl2_loss(labels, pred_masks)
        return composite_loss


class CompositeL2Loss(nn.Module):
    """Composite loss function"""    
    def __init__(self):
        super(CompositeL2Loss, self).__init__()
        #self.loss_fn_alex = lpips.LPIPS(net='alex', verbose = False).cuda()
        self.loss_fn_vgg = lpips.LPIPS(net='vgg', verbose = False).cuda()
        self.ssim_loss = ssim_loss
        self.psnr_loss = psnr_loss
        self.dwtl2_loss = DWTL2Loss()
        self.pixell2_loss = nn.MSELoss()

    def forward(self, labels, pred_masks):
        #composite_loss = 0.4 * self.loss_fn_alex(labels, pred_masks) + 0.1 * self.ssim_loss(labels, pred_masks, 11) + 0.2 * self.psnr_loss(labels, pred_masks, 1) + 0.3 * self.dwtl2_loss(labels, pred_masks)
        composite_loss = 0.6 * self.pixell2_loss(labels, pred_masks) + 0.4 * self.dwtl2_loss(labels, pred_masks)
        return composite_loss



class PixelL1Loss(nn.Module):
    """PixelWise L1 Loss as used in https://openaccess.thecvf.com/content/ACCV2020/papers/Chang_TinyGAN_Distilling_BigGAN_for_Conditional_Image_Generation_ACCV_2020_paper.pdf """
    def __init__(self):
        super(PixelL1Loss, self).__init__()
        self.l1loss = nn.L1Loss()

    def forward(self, labels, pred_masks):
        loss = self.l1loss(labels, pred_masks)
        return loss


class PixelL2Loss(nn.Module):
    """PixelWise L2 Loss"""
    def __init__(self):
        super(PixelL2Loss, self).__init__()
        self.l2loss = nn.MSELoss()

    def forward(self, labels, pred_masks):
        loss = self.l2loss(labels, pred_masks)
        return loss


class DWTL1Loss(nn.Module):
    """DWT L2 loss function"""    
    def __init__(self):
        super(DWTL1Loss, self).__init__()
        self.loss = nn.L1Loss().cuda()
        self.dwt = DWTForward(J=3, wave='db1', mode='zero').cuda()

    def forward(self, labels, pred_masks):
        (yl_labels, yh_labels) = self.dwt(labels)
        (yl_pred_masks, yh_pred_masks) = self.dwt(pred_masks)
        dwtl1_loss = self.loss(yh_labels[0], yh_pred_masks[0]) + self.loss(yh_labels[1], yh_pred_masks[1]) + self.loss(yh_labels[2], yh_pred_masks[2])
        return dwtl1_loss



# https://github.com/VainF/pytorch-msssim/

class CompositeLoss(nn.Module):
    """Composite loss function"""    
    def __init__(self):
        super(CompositeLoss, self).__init__()
        #self.loss_fn_alex = lpips.LPIPS(net='alex', verbose = False).cuda()
        #self.loss_fn_vgg = lpips.LPIPS(net='vgg', verbose = False).cuda()
        #self.ssim_loss = ssim_loss
        #self.psnr_loss = psnr_loss
        #self.dwtl1_loss = DWTL1Loss()
        self.pixell1_loss = PixelL1Loss()
        self.ffl = FFL(loss_weight=1.0, alpha=0.2)

    def forward(self, labels, pred_masks):
        #composite_loss = 0.4 * self.loss_fn_alex(labels, pred_masks) + 0.1 * self.ssim_loss(labels, pred_masks, 11) + 0.2 * self.psnr_loss(labels, pred_masks, 1) + 0.3 * self.dwtl2_loss(labels, pred_masks)
        #composite_loss = 0.3 * self.pixell1_loss(labels, pred_masks) + 0.2 * (1 - ms_ssim( labels, pred_masks, data_range=1, size_average=True )) + 0.5*self.ffl(pred_masks, labels)
        #composite_loss = self.ffl(pred_masks, labels)
        composite_loss = self.pixell1_loss(labels, pred_masks)
        return composite_loss