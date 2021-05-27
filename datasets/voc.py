import os
import torch
from torch.utils import data
import numpy as np
from skimage.io import imread
from PIL import Image, ImageDraw


class PascalVOC(data.Dataset):
    def __init__(self, path, images):
        
        self.path = path
        self.images = images
        self.images_path = os.path.join(path, 'JPEGImages')
        self.shapes_path = os.path.join(path, 'SegmentationClass')
        
    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        name = self.images[index]
        
        # load image
        image_path = os.path.join(self.images_path, name + ".jpg")
        image = imread(image_path)
        
        # load points
        shapes_path = os.path.join(self.shapes_path, name + ".png")
        shapes = imread(shapes_path)[:,:,:3]
        target = np.zeros((len(shapes), len(shapes[0])))
        
        for row in range(len(shapes)):
            for col in range(len(shapes[row])):
                if sum(shapes[row, col]) > 0:
                    target[row, col] = 1
                else:
                    pass
        
        item = {"images": image, 
                "shapes": target,
                "meta": {"index": index, "path": image_path}
        }
                
        return item