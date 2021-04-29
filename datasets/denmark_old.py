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
    def __init__(self, split, datadir, transform):
        self.split = split
        self.transform = transform
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
        self.shapes_path = os.path.join(datadir, 'shapes')
        self.cob_path = os.path.join(datadir, 'cob')

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, index):
        name = self.img_names[index]
        path = os.path.join(self.img_path, name + ".jpg")
        
        # LOAD IMG, POINT
        image = imread(path)
        #image = np.load(os.path.join(self.img_path, name + ".npy"))
        nrows, ncols = len(image), len(image[0]) 
        
        points = np.zeros((nrows, ncols, 1), dtype = int)
        points_path = os.path.join(self.points_path, name + "_points.npy")
        if os.path.isfile(points_path): 
            points_src = np.load(points_path)
            for point in points_src:
                w, h = point[0], point[1]
                points[h-1][w-1] = [1]
                
        shapes_path = os.path.join(self.shapes_path, name + "_shapes.npy")
        if os.path.isfile(shapes_path): 
            shapes = list(np.load(shapes_path, allow_pickle = True))
        else:
            shapes = []
                
        cob_path = os.path.join(self.cob_path, name + ".jpg")
        if os.path.isfile(cob_path):
            cob = imread(cob_path)[:, :, 1].astype('float64')
            cob /= 255.0
            #cob = 1-cob
        else:
            cob = None
        
        counts = torch.LongTensor(np.array([int(points.sum())]))   
        # ICH SOLLTE HIER NICHT DIE SHAPES ALS LISTE MITGEBEN.... BAD PRACTICE.... TRANSFORMER AUSEINANDERNEHMEN, SHAPE ALS BILD INTEGRIEREN, LOSS UMSTELLEN
        image, points = transformers.applyTransform(self.split, image, points, transform_name = self.transform)
        
        item = {"images": image, 
                "points": points.squeeze(),
                "shapes": shapes,
                "counts": counts, 
                "meta": {"index": index,
                         "path": path}}
        
        #if cob is not None:
            #item['cob'] = cob
            
        return item