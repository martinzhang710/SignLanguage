"""
@ Create on 01-05-2022

@ author: PENG ZHANG
"""

import warnings
warnings.filterwarnings('ignore')
import torch
import torch.nn as nn
from dataset import * 
from torchvision.models.resnet import resnet18
import torch.optim.lr_scheduler as lr_scheduler
import torch.optim as optim
from torch.autograd import Variable


device = "cuda:0"
data_path = "./image/train/labels.txt"
datasets = DataLoader(ImageData(data_path), batch_size=20, shuffle=True, num_workers=8)
net = resnet18().to(device)
net.fc = nn.Linear(512, 10).to(device)
crossentropy = nn.CrossEntropyLoss().to(device)
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0005)


net.train()
print("======== trian strat ==========")
for epoch in range(50):
    for image, label in datasets:
        image = Variable(image).to(device)
        label = Variable(label).to(device)
        
        output = net(image)
        _, predict = torch.max(output, axis=1)

        optimizer.zero_grad()
        loss = crossentropy(output, label)

        loss.backward()
        optimizer.step()
    print(f"epoch = {epoch}\tloss = {loss}")

torch.save(net, "net18_2.pt")

