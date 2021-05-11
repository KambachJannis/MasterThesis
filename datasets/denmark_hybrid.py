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
        self.points_path = os.path.join(path, 'points_'+object_type)
        self.shapes_path = os.path.join(path, 'shapes_'+object_type)
    
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
        
        # load points
        points_path = os.path.join(self.points_path, name + "_points.npy")
        points = np.zeros((n_rows, n_cols, 1), dtype = int)
        
        if os.path.isfile(points_path):
            points_list = np.load(points_path)
            label_point = 1
        else:
            points_list = []
            label_point = 0
        
        for point in points_list:
            x, y = point[0]-1, point[1]-1
            points[y][x] = [1]
            
        counts = torch.LongTensor(np.array([int(points.sum())]))
        points = torch.LongTensor(points).squeeze()
                
        # load shapes
        shapes_path = os.path.join(self.shapes_path, name + "_shapes.npy")
        shapes = Image.new('L', (n_rows, n_cols), 0)
        
        if os.path.isfile(points_path):
            shapes_list = list(np.load(shapes_path, allow_pickle = True))
            label_shape = 1
        else:
            shapes_list = []
            label_shape = 0
            
        for shape in shapes_list:
            flat_list = [item for sublist in shape for item in sublist]
            ImageDraw.Draw(shapes).polygon(flat_list, outline=1, fill=1)
        
        shapes = np.array(shapes)
        
        item = {"images": image, 
                "points": points,
                "counts": counts,
                "label_p": label_point,
                "shapes": shapes,
                "label_s": label_shape,
                "meta": {"index": index, "path": image_path}
        }
                
        return item