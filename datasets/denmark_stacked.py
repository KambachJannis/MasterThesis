# System
import os
# Torch
import torch
from torch.utils import data
# Data Handling
import numpy as np
# Image Handling
from skimage.io import imread
from PIL import Image, ImageDraw


class Denmark(data.Dataset):
    def __init__(self, path, images, object_type, n_classes, transform = None):
        
        self.path = path
        self.images = images
        self.transform = transform
        self.object_type = object_type
        self.n_classes = n_classes
        self.images_path = os.path.join(path, 'images')
        self.output_path = os.path.join(path, 'model_output_'+object_type)
    
    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        name = self.images[index]
        
        # load image
        image_path = os.path.join(self.images_path, name + ".jpg")
        image = imread(image_path)
        n_rows, n_cols = len(image), len(image[0]) 
        if self.transform is not None:
            image = self.transform(image)
                
        # load shapes
        shapes_path = os.path.join(self.output_path, name + ".npy")
        shapes = np.round(np.load(shapes_path, allow_pickle = True))
        
        item = {"images": image, 
                "shapes": shapes,
                "meta": {"index": index, "path": image_path}
        }
                
        return item