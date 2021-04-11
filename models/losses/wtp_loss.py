import torch
import numpy as np
import torch.nn.functional as F

def computeLoss(points, probs):
    """
    points: n x c x h x w
    probs: h x w (0 or 1)
    
    """
    loss = 0.
    # eliminate single dimensions (A x B x 1 -> A x B)
    points = points.squeeze()
    assert(points.max() <= 1)
    probs = probs.squeeze()
    # unfold A x B tensor to A*B x 1 tensor
    probs_flat = probs.view(-1)
    
    # get list of all relevant pixels and their GT labels
    checklist = getPixelChecklist(points, probs)
    for item in checklist:
        # get relevant pixels for this item
        item_ids = probs_flat[item['id_list']]
        # init tensor of same size and fill with label value
        item_labels = torch.ones(item_ids.shape, device = item_ids.device) * item['label']
        # essentially log function and output divided by #items (mean reduction)
        loss += item['scale'] * F.binary_cross_entropy(item_ids, item_labels, reduction='mean')
    
    # add objectness loss
    objectness = objectnessLoss(points, probs)
    print(f"base loss: {loss} - objectness: {objectness}")
    loss += objectness
    
    return loss

@torch.no_grad()
def getPixelChecklist(points, probs):
    """
    For each loss function part, builds a list of relevant pixels and their GT labels.
    
    """
    checklist = []
    
    ################ IMAGE LEVEL ######################
    pt_flat = points.view(-1)
    pr_flat = probs.view(-1)
    class_list = points.unique()
    
    if 0 in class_list:
        # pixel with highest chance for background class
        id_background = pr_flat.argmin() #returns ONE index
        checklist += [{'scale': 1, 'id_list': [id_background], 'label': 0}]
        
    if 1 in class_list:
        # pixel with highest chance for blob class
        id_foreground = pr_flat.argmax() # returns ONE index
        checklist += [{'scale': 1, 'id_list': [id_foreground], 'label': 1}]   

    ################ POINT LEVEL ######################
    if 1 in class_list:
        # locate point labels
        ids_labels = torch.where(pt_flat==1)[0]
        checklist += [{'scale': len(ids_labels), 'id_list': ids_labels, 'label': 1}]
        
    return checklist

def objectnessLoss(probs):
    objectness = getObjectness()
    objectness_flat = objectness.view(-1)
    pr_flat = probs.view(-1)
    n = len(pr_flat)
    
    ids_background = torch.where(pr_flat == 0)[0]
    ids_foreground = torch.where(pr_flat == 1)[0]
    
    score_background = torch.sum(torch.log(objectness_flat[ids_background]))
    score_foreground = torch.sum(torch.log(1 - objectness_flat[ids_foreground]))
    score = (score_background + score_foreground) / n
    
    return score
    
def getObjectness(): #MATLAB ENGINE MISSING
    pass