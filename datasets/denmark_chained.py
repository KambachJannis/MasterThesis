# System
import os
# Torch
import torch
from torch.utils import data
from torchvision import transforms
# Data Handling
import numpy as np
# Image Handling
from skimage.io import imread
from PIL import Image, ImageDraw
from skimage.morphology import label as ski_label
# Custom
import models

class Denmark(data.Dataset):
    def __init__(self, path, images, object_type, n_classes, transform = None):
        
        self.path = path
        self.images = images
        self.transform = transform
        self.object_type = object_type
        self.n_classes = n_classes
        self.images_path = os.path.join(path, 'images')
        self.model_output = []
        
        checkpoint = torch.load("/home/jovyan/work/runs/buildings_points_final/checkpoint_best.pth")
        self.model = models.getNet('vgg16', 2).cuda().load_state_dict(checkpoint['model']).eval()        
        
    
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
            
        # process image via model to get noisy targets
        if index not in self.model_output:
            with torch.no_grad():
                out = self.model.forward(image.cuda())
        
            probs = out.sigmoid().cpu().detach().numpy()
            shapes = ski_label((probs > 0.5).astype('uint8') == 1)
            self.model_output[index] = shapes
        else:
            shapes = self.model_output[index]
        
        item = {"images": image, 
                "shapes": shapes,
                "meta": {"index": index, "path": image_path}
        }
                
        return item