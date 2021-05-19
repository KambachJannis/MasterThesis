import torch
import torch.nn as nn
import torch.nn.functional as F
from losses import cob_loss, point_loss

def getLoss(loss_name):
    if loss_name == "BCELoss":
        criterion = nn.BCELoss()
    elif loss_name == "CrossEntropy":
        criterion = nn.CrossEntropyLoss()
    elif loss_name == "point_cob":
        criterion = cob_loss.computeLoss
    elif loss_name == "point":
        criterion = point_loss.computeLoss
    elif loss_name == "hybrid":
        criterion = point_loss.computeLoss # or COB Loss
    elif loss_name == "dice":
        criterion = DiceLoss()
    else:
        raise ValueError(f"Loss {loss_name} not implemented.")

    return criterion

class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        #inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.reshape(-1)
        targets = targets.reshape(-1)
        
        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        
        return 1 - dice