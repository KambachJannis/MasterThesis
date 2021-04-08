# Torch
import torch
from torch.utils import data
import torchvision.transforms.functional as FT
# Data Handling
import numpy as np
# Image Handling
import fiona
import rasterio
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

        self.img_path = os.path.join(datadir, 'images')
        self.point_path = os.path.join(datadir, 'annotations/85blocks_trees.shp')
        self.img_names = [name.replace(".tif\n","").replace(".tif","") for name in utils.readText(fname)]
        print(self.img_names)

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, index):
        name = self.img_names[index]
      
        # oldscool reading
        #image = imread(os.path.join(self.img_path, name + ".tif"))
        #image = np.delete(image, 3, 2)
        
        src = rasterio.open(os.path.join(self.img_path, name + ".tif"))
        # RGB image with first 3 bands
        image = np.stack((src.read(1), src.read(2), src.read(3)), axis = 2)
        # load points in this area
        bounds = list(src.bounds)
        points = loadPoints(self, bounds)[:,:,:1].clip(0,1)
        counts = torch.LongTensor(np.array([int(points.sum())]))   
       
        image, points = transformers.applyTransform(self.split, image, points, transform_name = self.exp_dict['dataset']['transform'])
            
        return {"images":image, 
                "points":points.squeeze(), 
                "counts":counts, 
                'meta':{"index":index}}
    
def loadPoints(self, bounds):
    with fiona.open(self.point_path) as shapefile:
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