import os
import random
import torch
import h5py
import cv2
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as F
from PIL import Image, ImageFilter, ImageDraw, ImageStat



class listDataset(Dataset):
    def __init__(self, root, shape=None, shuffle=True, transform=None,  train=False):
        if train:
            root = root * 4
        random.shuffle(root)
        
        self.nSamples = len(root)
        self.lines = root
        self.transform = transform
        self.train = train
        self.shape = shape
        
        
    def __len__(self):
        return self.nSamples
    def __getitem__(self, index):
        assert index <= len(self), 'index range error' 
        
        img_path = self.lines[index]
        
        img,target,dots, f_name = load_data(img_path,self.train)
        
        if self.transform is not None:
            img = self.transform(img)
        return img, target, dots, f_name



def load_data(img_path,train = True):
    gt_path = img_path.replace('.png','.h5').replace('images','ground_truth')
    dot_path = img_path.replace('.png', '.txt').replace('cell', 'dots').replace('images','ground_truth')
    file_name = img_path.split('.png')[0].split('/')[-1][:-4]
    
    img = Image.open(img_path).convert('RGB')
    gt_file = h5py.File(gt_path, 'r')
    target = np.asarray(gt_file['density'])
    with open(dot_path) as f:
        res = np.array([i.strip('\n') for i in f.readlines()])
        dots = [[float(i.split(',')[0]),float(i.split(',')[1])] for i in res]
        dots = np.array(dots)
    
    return img, target, dots, file_name