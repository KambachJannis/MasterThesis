import os
import torch
from torch.utils import data
import numpy as np
from skimage.io import imread


class Trancos(data.Dataset):
    def __init__(self, path, images, object_type, n_classes, transform = None):
        
        self.path = os.path.join(path, 'images')
        self.images = images
        self.transform = transform
        self.object_type = object_type
        self.n_classes = n_classes

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        name = self.images[index]

        # LOAD IMG, POINT, and ROI
        image = imread(os.path.join(self.path, name + ".jpg"))
        points = imread(os.path.join(self.path, name + "dots.png"))[:,:,:1].clip(0,1)
        if self.transform is not None:
            image = self.transform(image)

        counts = torch.LongTensor(np.array([int(points.sum())]))
        points = torch.LongTensor(points).squeeze()
            
        return {"images":image, 
                "points":points, 
                "counts":counts, 
                'meta':{"index":index}}
