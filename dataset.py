"""
@ Create on 01-05-2022 

@ author: PENG ZHANG
"""

import torch

import numpy as np
import os
from PIL import Image
import cv2
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class ImageData(Dataset):
    def __init__(self, root, training=True):
        super().__init__()
        self.images, self.labels = [], []
        with open(root, "r") as f:
            file_names = f.readlines()
            for file_name in file_names:
                image_name = file_name.split()[0]
                label_name = int(file_name.split()[1])
                self.images.append(image_name)
                self.labels.append(label_name)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        label = torch.tensor(self.labels[index])
        images = self.images[index]
        transfor = transforms.Compose([
            lambda image : Image.open(image).convert("RGB"),
            transforms.ToTensor(),
            transforms.Normalize([0.506, 0.521, 0.535], [0.098, 0.096, 0.063])])
        image = transfor(images)
        return image, label




