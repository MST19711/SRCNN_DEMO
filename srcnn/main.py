import imp
from importlib.resources import path
from importlib.util import module_for_loader
from pkgutil import ImpImporter
from re import I, S
from turtle import forward
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda, Compose
import os
import cv2 as cv
import MyDataset
import MyModel

Device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

f_1 = int(9)
f_2 = int(1)
f_3 = int(5)
n_1 = int(64)
n_2 = int(32)

trainning_data_path = input("Trainning_data_path : ")
model_path = input("Model path : ")
model_name = input("Model name : ")
if not os.path.isdir(model_path):
    model = torch.load(model_path)
    model_path = input("Model save path : ")
else:
    model = MyModel.Model(f_1,f_2,f_3,n_1,n_2).to(Device)

trainning_data = MyDataset.imgs(trainning_data_path)
loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(),lr = 0.001)
Batch_size = int(64)
trainning_dataloader = DataLoader(trainning_data,batch_size = Batch_size,shuffle = True)
Epach = int(1000)
for i in range(Epach):
    print("Epach : " + str(i))
    for batch,(In,traget) in enumerate(trainning_dataloader):
        In,target = In.to(Device),target.to(Device)
        SR_imgs = model(In)
        loss = loss_fn(SR_imgs,target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print("    Batch : " + str(batch) + "  loss : " + loss.item())
    if Epach % 5 == 0:
        torch.save(model,model_path + "\\" + model_name + "_" + str(i+1))