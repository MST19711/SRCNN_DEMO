import imp
from importlib.resources import path
from importlib.util import module_for_loader
from pkgutil import ImpImporter
from re import I, S
from statistics import mode
from turtle import forward
import numpy
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

save_test_img = input("Test imgs save path : ")
test_img_path = input("Test imgs path : ")
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
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.5, 0.999))
Batch_size = int(64)
trainning_dataloader = DataLoader(trainning_data,batch_size = Batch_size,shuffle = True)
Epach = int(100)
for i in range(Epach):
    print("Epach : " + str(i))
    for batch,(In,target) in enumerate(trainning_dataloader):
        In,target = In.to(Device),target.to(Device)
        SR_imgs = model(In)
        loss = loss_fn(SR_imgs,target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if batch % 10 == 0:
            print("    Batch : " + str(batch) + "  loss : " + str(loss))
    #if i % 5 == 0:
    torch.save(model,model_path + "\\" + model_name + "_" + str(i+1))
    test_img = cv.imread(test_img_path)
    test_img = cv.resize(test_img,dsize = (0,0),fx = 2,fy = 2,interpolation = cv.INTER_CUBIC)
    test_input = numpy.zeros((1,3,test_img.shape[0],test_img.shape[1]))
    test_input[0] = numpy.transpose(test_img,(2,0,1))
    test_input = torch.from_numpy(test_input).to(Device).float()
    test_out = model(test_input)
    test_out = test_out.to("cpu")
    out_img = test_out[0].detach().numpy()
    out_img = numpy.transpose(out_img,(1,2,0))
    cv.imwrite(save_test_img + "\\" + model_name + "_Epach_" + str(i) + ".png",out_img)
