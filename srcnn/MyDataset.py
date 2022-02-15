from importlib.resources import path
from importlib.util import module_for_loader
from re import I, S
from turtle import forward
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda, Compose
import os
import cv2 as cv


class imgs(Dataset):
    def __init__(self,path) -> None:
        super().__init__()
        self.filelist = []
        for i in os.walk(path):
            if not os.path.isdir(i):
                self.filelist.append(path + "\\" + i)
    
    def __len__(self):
        return len(self.filelist)

    def __getitem__(self, index):
        target = cv.imread(self.filelist[index])
        train = target
        train = cv.resize(train,fx = 0.5,fy = 0.5)
        train = cv.resize(train,fx = 2,fy = 2,interpolation = cv.INTER_CUBIC)
        target = torch.from_numpy(target)
        train = torch.from_numpy(target)
        return (train,target)