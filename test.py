"""
@ Create on 01-05-2022

@ author: PENG ZHANG
"""

import warnings
warnings.filterwarnings('ignore')
import torch
from PIL import Image
from dataset import *
from torchvision.models.resnet import resnet18


class ImageData(Dataset):
    def __init__(self, image, training=True):
        super().__init__()
        self.images = [image]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        images = self.images[index]
        transfor = transforms.Compose([
            lambda image : Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)),
            transforms.ToTensor(),
            transforms.Normalize([0.506, 0.521, 0.535], [0.098, 0.096, 0.063])])
        image = transfor(images)
        return image

def gtest(net, image):
    datasets = DataLoader(ImageData(image), batch_size=1)
    for image in datasets:
        output = net(image.cuda())
        result = torch.argmax(output, axis=1)
        if int(result[0]) == 0:
            return 10
        else:
            return int(result[0])

if __name__ == "__main__":
    net = torch.load("./net18.pt").to("cuda:0")
    net.eval()
    import os
    root = "./image/train/1/"
    for image_name in os.listdir(root):
        image_path = root + image_name
        image = cv2.imread(image_path)
        print(gtest(net, image))
        break


