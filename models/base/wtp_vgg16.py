
import torch.nn as nn
import torchvision
import torch
from skimage import morphology as morph
import numpy as np

import torch.utils.model_zoo as model_zoo

# Note that the original Caffe Model has different multipliers for the LR of weights and biases
class WTP_VGG16(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.n_classes = n_classes
        # PREDEFINE LAYERS
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        self.relu = nn.ReLU(inplace=True)
      
        # VGG16 PART
        self.conv1_1 = conv3x3(3, 64, stride=1, padding=100)
        self.conv1_2 = conv3x3(64, 64)
        
        self.conv2_1 = conv3x3(64, 128)
        self.conv2_2 = conv3x3(128, 128)
        
        self.conv3_1 = conv3x3(128, 256)
        self.conv3_2 = conv3x3(256, 256)
        self.conv3_3 = conv3x3(256, 256)

        self.conv4_1 = conv3x3(256, 512)
        self.conv4_2 = conv3x3(512, 512)
        self.conv4_3 = conv3x3(512, 512)

        self.conv5_1 = conv3x3(512, 512)
        self.conv5_2 = conv3x3(512, 512)
        self.conv5_3 = conv3x3(512, 512)
        
        self.fc6 = nn.Conv2d(512, 4096, kernel_size=7, stride=1, padding=0)
        self.dropout = nn.Dropout()
        self.fc7 = nn.Conv2d(4096, 4096, kernel_size=1, stride=1, padding=0)
        
        # UPSAMPLING PATH
        self.scorefr = nn.Conv2d(4096, self.n_classes, kernel_size=1, stride=1, padding=0)
        self.scorefr.weight.data.zero_()
        self.scorefr.bias.data.zero_()
        self.upsample = nn.ConvTranspose2d(self.n_classes, self.n_classes, kernel_size=64, stride=32)
        self.upsample.weight.data.copy_(get_upsampling_weight(self.n_classes, self.n_classes, 64))
        
        # Pretrained layers
        pth_url = 'https://download.pytorch.org/models/vgg16-397923af.pth' # download from model zoo
        state_dict = model_zoo.load_url(pth_url)
        layer_names = [layer_name for layer_name in state_dict]
        counter = 0
        
        for p in self.parameters():
            if counter < 26:  # conv1_1 to pool5
                p.data = state_dict[ layer_names[counter] ]
            elif counter == 26:  # fc6 weight
                p.data = state_dict[ layer_names[counter] ].view(4096, 512, 7, 7)
            elif counter == 27:  # fc6 bias
                p.data = state_dict[ layer_names[counter] ]
            elif counter == 28:  # fc7 weight
                p.data = state_dict[ layer_names[counter] ].view(4096, 4096, 1, 1)
            elif counter == 29:  # fc7 bias
                p.data = state_dict[ layer_names[counter] ]
                
            counter += 1
        

    def forward(self, x):
        n,c,h,w = x.size()
        # VGG16 PART
        conv1_1 =  self.relu(  self.conv1_1(x) )
        conv1_2 =  self.relu(  self.conv1_2(conv1_1) )
        pool1 = self.pool(conv1_2)
        
        conv2_1 =  self.relu(   self.conv2_1(pool1) )
        conv2_2 =  self.relu(   self.conv2_2(conv2_1) )
        pool2 = self.pool(conv2_2)
        
        conv3_1 =  self.relu(   self.conv3_1(pool2) )
        conv3_2 =  self.relu(   self.conv3_2(conv3_1) )
        conv3_3 =  self.relu(   self.conv3_3(conv3_2) )
        pool3 = self.pool(conv3_3)
        
        conv4_1 =  self.relu(   self.conv4_1(pool3) )
        conv4_2 =  self.relu(   self.conv4_2(conv4_1) )
        conv4_3 =  self.relu(   self.conv4_3(conv4_2) )
        pool4 = self.pool(conv4_3)
        
        conv5_1 =  self.relu(   self.conv5_1(pool4) )
        conv5_2 =  self.relu(   self.conv5_2(conv5_1) )
        conv5_3 =  self.relu(   self.conv5_3(conv5_2) )
        pool5 = self.pool(conv5_3)
        
        fc6 = self.dropout( self.relu(   self.fc6(pool5) ) )
        fc7 = self.dropout( self.relu(   self.fc7(fc6) ) )
        
        # SEMANTIC SEGMENTATION PART
        score = self.scorefr( fc7 )
        bigscore = self.upsample(score)

        return bigscore[:, :, 31: (31 + h), 31: (31 + w)].contiguous() #todo: check matrix dimensions


def get_upsampling_weight(in_channels, out_channels, kernel_size):
    """Make a 2D bilinear kernel suitable for upsampling"""
    factor = (kernel_size + 1) // 2
    
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    
    og = np.ogrid[:kernel_size, :kernel_size]
    weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size), dtype=np.float64)
    
    weight[range(in_channels), range(out_channels), :, :] = (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)
    
    return torch.from_numpy(weight).float()    
    
def conv3x3(in_planes, out_planes, stride=1, padding=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=(3,3), stride=(stride,stride),
                     padding=(padding,padding))

def conv1x1(in_planes, out_planes, stride=1):
    "1x1 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                     padding=0)