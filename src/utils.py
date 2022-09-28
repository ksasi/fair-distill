import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, Dataset
import glob
import cv2
import albumentations as A
import numpy as np


class RESIDEDataset(Dataset):
    def __init__(self, path, train = True, transform=None, transform_input=None, transform_target=None):
        self.path = path
        self.transform = transform
        self.transform_input=transform_input
        self.transform_target=transform_target
        self.train = train
        if self.train:
          self.images_hazy = sorted([file for file in glob.glob(self.path + 'hazy/' + '*')])
          self.images_clear = sorted([file for file in glob.glob(self.path + 'clear/' + '*')])
          self.clear_base_path = self.path + 'clear/'
        else:
          self.images_hazy = sorted([file for file in glob.glob(self.path + 'hazy/' + '*')])
          self.images_clear = sorted([file for file in glob.glob(self.path + 'gt/' + '*')])
          self.clear_base_path = self.path + 'gt/'

            
    def __getitem__(self,index):
        image_hazy_path = self.images_hazy[index]
        #hazy = Image.open(image_hazy_path)
        hazy = cv2.imread(image_hazy_path)
        hazy = cv2.cvtColor(hazy, cv2.COLOR_BGR2RGB)

        #print(image_hazy_path)
        clear_hazy_path = self.clear_base_path + image_hazy_path.split('/')[-1].split('_')[0] + '.png'
        #print(clear_hazy_path)
        #clear = Image.open(clear_hazy_path)
        clear = cv2.imread(clear_hazy_path)
        clear = cv2.cvtColor(clear, cv2.COLOR_BGR2RGB)


        if self.transform:
            #transformed = self.transform(image=hazy, mask=clear)
            hazy_data = self.transform(image=hazy)
            #hazy_transformed = transformed['image']
            hazy_transformed_base = hazy_data['image']
            input_transformed = self.transform_input(image=hazy_transformed_base)
            hazy_transformed = input_transformed['image']
            clear_data = A.ReplayCompose.replay(hazy_data['replay'], image=clear)
            #clear_transformed = torch.squeeze(transformed['mask']).permute(2,0,1)
            clear_transformed_base = clear_data['image']
            target_transformed = self.transform_target(image=clear_transformed_base)
            clear_transformed = target_transformed['image']

        return hazy_transformed, clear_transformed
        
    def __len__(self):
        return len(self.images_hazy)




class SANDDataset(Dataset):
    def __init__(self, path):
        self.path = path
        '''
        self.transform = transform
        self.transform_input=transform_input
        self.transform_target=transform_target
        '''
        self.images_sandy = sorted([file for file in glob.glob(self.path + 'source/' + '*')])
        self.images_clear = sorted([file for file in glob.glob(self.path + 'target/' + '*')])
        ####self.clear_base_path = self.path + 'source/'
        self.clear_base_path = self.path + 'target/'

            
    def __getitem__(self,index):
        image_sandy_path = self.images_sandy[index]
        #sandy = Image.open(image_sandy_path)
        sandy = cv2.imread(image_sandy_path)
        sandy = cv2.cvtColor(sandy, cv2.COLOR_BGR2RGB)

        #print(image_sandy_path)
        clear_sandy_path = self.clear_base_path + image_sandy_path.split('/')[-1].split('_')[0]
        #print(clear_sandy_path)
        #clear = Image.open(clear_sandy_path)
        clear = cv2.imread(clear_sandy_path)
        clear = cv2.cvtColor(clear, cv2.COLOR_BGR2RGB)

        '''
        if self.transform:
            #transformed = self.transform(image=sandy, mask=clear)
            sandy_data = self.transform(image=sandy)
            #sandy_transformed = transformed['image']
            sandy_transformed_base = sandy_data['image']
            input_transformed = self.transform_input(image=sandy_transformed_base)
            sandy_transformed = input_transformed['image']
            clear_data = A.ReplayCompose.replay(sandy_data['replay'], image=clear)
            #clear_transformed = torch.squeeze(transformed['mask']).permute(2,0,1)
            clear_transformed_base = clear_data['image']
            target_transformed = self.transform_target(image=clear_transformed_base)
            clear_transformed = target_transformed['image']
        '''
        sandy_transformed = sandy
        clear_transformed = clear
        return sandy_transformed, clear_transformed
        
    def __len__(self):
        return len(self.images_sandy)


class WrapperDataset(Dataset):
    def __init__(self, dataset, transform=None, transform_input=None, transform_target=None):
        self.dataset = dataset
        self.transform = transform
        self.transform_input=transform_input
        self.transform_target=transform_target

    def __getitem__(self, index):
        sandy, clear = self.dataset[index]
        if self.transform:
            sandy_data = self.transform(image=sandy)
            #sandy_transformed = transformed['image']
            sandy_transformed_base = sandy_data['image']
            input_transformed = self.transform_input(image=sandy_transformed_base)
            sandy_transformed = input_transformed['image']
            clear_data = A.ReplayCompose.replay(sandy_data['replay'], image=clear)
            #clear_transformed = torch.squeeze(transformed['mask']).permute(2,0,1)
            clear_transformed_base = clear_data['image']
            target_transformed = self.transform_target(image=clear_transformed_base)
            clear_transformed = target_transformed['image']
        return sandy_transformed, clear_transformed

    def __len__(self):
        return len(self.dataset)




class DISTILLEDDataset(Dataset):
    def __init__(self, path, transform=None):
        self.path = path
        self.transform = transform
        #self.transform_input=transform_input
        #self.transform_target=transform_target
        #self.train = train
        #if self.train:
        self.images = sorted([file for file in glob.glob(self.path + 's_images/' + '*')])
        self.latents = sorted([file for file in glob.glob(self.path + 's_latentsz/' + '*')])

            
    def __getitem__(self,index):
        image_path = self.images[index]
        #hazy = Image.open(image_hazy_path)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.transform(image)
        latent_path = self.latents[index]
        latent = np.load(latent_path)
        latent = torch.from_numpy(latent)

        '''
        if self.transform:
            #transformed = self.transform(image=hazy, mask=clear)
            hazy_data = self.transform(image=hazy)
            #hazy_transformed = transformed['image']
            hazy_transformed_base = hazy_data['image']
            input_transformed = self.transform_input(image=hazy_transformed_base)
            hazy_transformed = input_transformed['image']
            clear_data = A.ReplayCompose.replay(hazy_data['replay'], image=clear)
            #clear_transformed = torch.squeeze(transformed['mask']).permute(2,0,1)
            clear_transformed_base = clear_data['image']
            target_transformed = self.transform_target(image=clear_transformed_base)
            clear_transformed = target_transformed['image']
        '''
        return latent, image
        
    def __len__(self):
        return len(self.images)