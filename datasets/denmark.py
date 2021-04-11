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
        self.n_patches = 40
        self.tile_width = 250
        self.tile_height = 250
        
        if split == "train":
            fname = os.path.join(datadir, 'image_sets', 'training.txt')

        elif split == "val":
            fname = os.path.join(datadir, 'image_sets', 'validation.txt')

        elif split == "test":
            fname = os.path.join(datadir, 'image_sets', 'test.txt')
        
        # RASTERIO MASK mit Overlap 
        self.img_path = os.path.join(datadir, 'images')
        self.point_path = os.path.join(datadir, 'annotations/85blocks_trees.shp')
        
        collection = []
        for name in utils.readText(fname):
            name_clean = name.replace(".tif\n","").replace(".tif","")
            collection.extend([name_clean + '_' + str(n) for n in list(range(self.n_patches**2))])
        
        self.img_names = collection
        print(self.img_names)

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, index):
        parts = self.img_names[index].split("_")
        filename = '_'.join(parts[:-1])
        tile_nr = parts[-1]
        
        # open with RasterIO
        src = rasterio.open(os.path.join(self.img_path, filename + ".tif"), window = window)
        
        # get window based on tile_nr
        ncols, nrows = src.meta['width'], src.meta['height']
        big_window = rio.windows.Window(col_off = 0, row_off = 0, width = ncols, height = nrows)
        
        col, row = tile_nr % self.n_patches, tile_nr // self.n_patches
        h_overlap = ((self.n_patches * self.tile_width) - ncols) / (self.n_patches - 1)
        v_overlap = ((self.n_patches * self.tile_height) - nrows) / (self.n_patches - 1)
        col_off, row_off = col * (self.tile_width - v_overlap), row * (self.tile_height - h_overlap)
        
        window = rio.windows.Window(col_off = col_off, row_off = row_off, 
                                    width = self.tile_height, height = self.tile_height).intersection(big_window)
        transform = rio.windows.transform(window, src.transform)
        
        # RGB image
        image = np.transpose(src.read(window = window)[:3])
        # load points in this area
        points = loadPoints(self, src)[:,:,:1].clip(0,1)
        counts = torch.LongTensor(np.array([int(points.sum())]))   
       
        image, points = transformers.applyTransform(self.split, image, points, transform_name = self.exp_dict['dataset']['transform'])
            
        return {"images":image, 
                "points":points.squeeze(), 
                "counts":counts, 
                'meta':{"index":index}}
    
def loadPoints(self, bounds):
    # SHAPEFILE Selber in bounds einschränken
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

def getTiles(src, width, height, vertical, horizontal):
    ncols, nrows = src.meta['width'], src.meta['height']
    big_window = rio.windows.Window(col_off = 0, row_off = 0, width = ncols, height = nrows)
    
    h_overlap = ((horizontal * width) - ncols) / (horizontal - 1)
    v_overlap = ((vertical * height) - nrows) / (vertical - 1)
    
    for y in range(vertical):
        row_off = y * (height - h_overlap)
        for x in range(horizontal):
            col_off = x * (width - v_overlap)
            window = rio.windows.Window(col_off = col_off, row_off = row_off, width = width, height = height).intersection(big_window)
            transform = rio.windows.transform(window, src.transform)
            yield window, transform

def generateMasks(self, src):
    pass