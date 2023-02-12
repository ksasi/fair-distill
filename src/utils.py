import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, Dataset
import glob
import cv2
import albumentations as A
import numpy as np
import dlib

class DISTILLEDDataset(Dataset):
    def __init__(self, path, transform=None):
        self.path = path
        self.transform = transform
        self.images = sorted([file for file in glob.glob(self.path + 's_images/' + '*')])
        self.latents = sorted([file for file in glob.glob(self.path + 's_latentsz/' + '*')])

    def detect_face(self, image_path, default_max_size=800,size = 300, padding = 0.25):
        cnn_face_detector = dlib.cnn_face_detection_model_v1('/workspace/FairFace/dlib_models/mmod_human_face_detector.dat')
        sp = dlib.shape_predictor('/workspace/FairFace/dlib_models/shape_predictor_5_face_landmarks.dat')
        base = 2000  # largest width and height
        img = dlib.load_rgb_image(image_path)

        old_height, old_width, _ = img.shape

        if old_width > old_height:
            new_width, new_height = default_max_size, int(default_max_size * old_height / old_width)
        else:
            new_width, new_height =  int(default_max_size * old_width / old_height), default_max_size
        img = dlib.resize_image(img, rows=new_height, cols=new_width)

        dets = cnn_face_detector(img, 1)
        num_faces = len(dets)
        if num_faces == 0:
            return img
        faces = dlib.full_object_detections()
        for detection in dets:
            rect = detection.rect
            faces.append(sp(img, rect))
        image = dlib.get_face_chips(img, faces, size=size, padding = padding)[0]
        return image

            
    def __getitem__(self,index):
        image_path = self.images[index]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.transform(image)
        latent_path = self.latents[index]
        latent = np.load(latent_path)
        latent = torch.from_numpy(latent)
        return latent, image
        
    def __len__(self):
        return len(self.images)