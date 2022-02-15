from importlib.util import module_for_loader
from turtle import forward
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda, Compose


class Model(nn.Module):
    def __init__(self,f_1,f_2,f_3,n_1,n_2) -> None:
        super(Model,self).__init__()
        self.n_1 = n_1
        self.n_2 = n_2
        self.f_1 = f_1
        self.f_2 = f_2
        self.f_3 = f_3
        self.SRCNN = nn.Sequential(
            nn.Conv2d(3,self.n_1,self.f_1,padding = int((self.f_1)/2)),
            nn.ReLU(),
            nn.Conv2d(self.n_1,self.n_2,self.f_2,padding = int((self.f_2)/2)),
            nn.ReLU(),
            nn.Conv2d(self.n_2,3,self.f_3,padding = int(self.f_3/2))
        )

    def forward(self,x):
        out = self.SRCNN(x)
        return out