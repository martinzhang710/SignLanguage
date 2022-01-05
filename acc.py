"""
@ Create on 01-05-2022 

@ author: PENG ZHANG
"""

import torch
from dataset import *
from tqdm import tqdm
from torchvision.models.resnet import resnet18


data_path = "./image/test/labels.txt"
dataloader = DataLoader(ImageData(data_path), batch_size=20, shuffle=True, num_workers=8)


net = torch.load("./net18.pt").cuda()
net.eval()

with torch.no_grad():
    acc_net, total = 0, 0
    for i, (images, labels) in tqdm(enumerate(dataloader, 0), total=len(dataloader)):
        images = images.cuda()
        labels = labels.cuda()

        result_net = net(images)

        result_net_prob = torch.argmax(result_net, axis=1)

        acc_net += (result_net_prob == labels).sum()
        total += len(labels)

    print("acc_net = ", acc_net / total)

