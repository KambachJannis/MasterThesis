# Torch
import torch
from torch.utils import data
import torchvision.transforms.functional as FT
# Data Handling
import numpy as np
# Image Handling
import fiona
import rasterio
from scipy.io import loadmat
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

        self.path = os.path.join(datadir, 'images')
        self.img_names = [name.replace(".jpg\n","") for name in utils.readText(fname)]
        print(self.img_names)

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, index):
        name = self.img_names[index]

        # LOAD IMG, POINT, and ROI
        image = imread(os.path.join(self.path, name + ".jpg"))
        #image = np.delete(image, 3, 2)
        points = loadPoints(self, name)[:,:,:1].clip(0,1)
        
        counts = torch.LongTensor(np.array([int(points.sum())]))   
        
        collection = list(map(FT.to_pil_image, [image, points]))
        image, points = transformers.applyTransform(self.split, image, points, transform_name = self.exp_dict['dataset']['transform'])
            
        return {"images":image, 
                "points":points.squeeze(), 
                "counts":counts, 
                'meta':{"index":index}}
    
def loadPoints(self, name):
    src = rasterio.open(os.path.join(self.path + "/tif/", name + ".tif"))
    bounds = list(src.bounds)
    
    with fiona.open(os.path.join(self.path, "tif/85blocks_trees.shp")) as shapefile:
        features = [feature["geometry"] for feature in shapefile]
    
    points = []
    for point in features:
        coords = list(point['coordinates'][:2])
        if (bounds[0] < coords[0] < bounds[2]) and (bounds[1] < coords[1] < bounds[3]):
            coords[0] = round((coords[0] - bounds[0])*8)
            coords[1] = round((coords[1] - bounds[1])*8)
            points.append(coords)
        
    dots = np.zeros((8000, 8000, 3), dtype = int)
    for point in points:
        w = point[0]
        h = point[1]
        dots[8000-h][w] = [255, 0, 0]
        
    return dots