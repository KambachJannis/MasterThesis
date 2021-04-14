# Torch
import torch
from torch.utils import data
import torchvision.transforms.functional as FT
# Data Handling
import numpy as np
# Image Handling
from skimage.io import imread
# Custom
from helpers import utils
from helpers import transformers
# System
import os

class Denmark(data.Dataset):
    def __init__(self, split, datadir, exp_dict):
        self.split = split
        self.exp_dict = exp_dict
        self.n_classes = 1
        
        if split == "train":
            fname = os.path.join(datadir, 'image_sets', 'training.txt')

        elif split == "val":
            fname = os.path.join(datadir, 'image_sets', 'validation.txt')

        elif split == "test":
            fname = os.path.join(datadir, 'image_sets', 'test.txt')
        
        self.img_names = [name.replace(".jpg\n","").replace(".jpg","") for name in utils.readText(fname)]
        self.img_path = os.path.join(datadir, 'images')
        self.points_path = os.path.join(datadir, 'points')

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, index):
        name = self.img_names[index]
        
        # LOAD IMG, POINT
        image = imread(os.path.join(self.img_path, name + ".jpg"))
        #image = np.load(os.path.join(self.img_path, name + ".npy"))
        nrows, ncols = len(image), len(image[0]) 
        
        points = np.zeros((nrows, ncols, 1), dtype = int)
        points_path = os.path.join(self.points_path, name + "_points.npy")
        if os.path.isfile(points_path): 
            points_src = np.load(points_path)
            for point in points_src:
                w, h = point[0], point[1]
                points[h-1][w-1] = [1]
        
        counts = torch.LongTensor(np.array([int(points.sum())]))   
       
        image, points = transformers.applyTransform(self.split, image, points, transform_name = self.exp_dict['dataset']['transform'])
            
        return {"images":image, 
                "points":points.squeeze(), 
                "counts":counts, 
                'meta':{"index":index,
                        "path":os.path.join(self.img_path, name + ".jpg")}}