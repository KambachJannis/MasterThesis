# Torch
import torch
from torch.utils import data
import torchvision.transforms.functional as FT
# Data Handling
import numpy as np
# Image Handling
from scipy.io import loadmat
from skimage.io import imread
# Custom
from helpers import utils
from helpers import transformers
# System
import os

class Trancos(data.Dataset):
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

        self.img_names = [name.replace(".jpg\n","") for name in utils.readText(fname)]
        self.path = os.path.join(datadir, 'images')

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, index):
        name = self.img_names[index]

        # LOAD IMG, POINT, and ROI
        image = imread(os.path.join(self.path, name + ".jpg"))
        points = imread(os.path.join(self.path, name + "dots.png"))[:,:,:1].clip(0,1)
        roi = loadmat(os.path.join(self.path, name + "mask.mat"))["BW"][:,:,np.newaxis]
        
        # LOAD IMG AND POINT
        image = image * roi
        image = shrink2RoI(image, roi)
        points = shrink2RoI(points, roi).astype("uint8")

        counts = torch.LongTensor(np.array([int(points.sum())]))   
        
        #unused variable
        #collection = list(map(FT.to_pil_image, [image, points]))
        image, points = transformers.applyTransform(self.split, image, points, transform_name = self.exp_dict['dataset']['transform'])
            
        return {"images":image, 
                "points":points.squeeze(), 
                "counts":counts, 
                'meta':{"index":index}}

def shrink2RoI(img, roi):
    """[summary]
    Parameters
    ----------
    img : [type]
        [description]
    roi : [type]
        [description]
    Returns
    -------
    [type]
        [description]
    """
    ind = np.where(roi != 0)
    y_min = min(ind[0])
    y_max = max(ind[0])
    x_min = min(ind[1])
    x_max = max(ind[1])

    return img[y_min:y_max, x_min:x_max]