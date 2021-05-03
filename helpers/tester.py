import torch
import numpy as np
from tqdm.notebook import tqdm


def testModel(model, test_loader, metric_list):
    
    results_dict = {}
    
    if 'mIoU' in metric_list:
        results_dict['mIoU'] = testMIoU(model, test_loader)
    if 'Dice' in metric_list:
        results_dict['Dice'] = testDice(model, test_loader)
    if 'pAccuracy' in metric_list:
        results_dict['pAccuracy'] = testAccuracy(model, test_loader)
        
    return results_dict


def testMIoU(model, test_loader, threshold = 0.5):
    
    model.eval()
    score_list = []
    
    for batch in tqdm(test_loader):
        # Load Data to GPU
        images = batch["images"].cuda()
        shapes = batch["shapes"].long().cuda()
        # Forward Prop
        logits = model.forward(images)
        probs = logits.sigmoid()
        # get mIoU score (the cutting is done bc unet returns [b x c x h x w] with c = 2 (background = 0, object = 1))
        score = calculateMIoU(probs.cpu().numpy()[:,1,...], shapes.cpu().numpy(), 1, threshold)
        # add to list
        score_list.append(score)
        
    return np.mean(score_list)


def calculateMIoU(probs, target, class_id, threshold):
    
    mask_preds = probs > threshold
    mask_target = target == class_id
    
    intersection = np.logical_and(mask_preds, mask_target).sum()
    union = np.logical_or(mask_preds, mask_target).sum()
    score = intersection / union
    
    return score


def testDice(model, test_loader, threshold = 0.5):
    
    model.eval()
    score_list = []
    
    for batch in tqdm(test_loader):
        # Load Data to GPU
        images = batch["images"].cuda()
        shapes = batch["shapes"].long().cuda()
        # Forward Prop
        logits = model.forward(images)
        probs = logits.sigmoid()
        # get mIoU score (the cutting is done bc unet returns [b x c x h x w] with c = 2 (background = 0, object = 1))
        score = calculateDice(probs.cpu().numpy()[:,1,...], shapes.cpu().numpy(), 1, threshold)
        # add to list
        score_list.append(score)
        
    return np.mean(score_list)


def calculateDice(probs, target, class_id, threshold):
    
    mask_preds = probs > threshold
    mask_target = target == class_id
    
    intersection = np.logical_and(mask_preds, mask_target).sum()
    total = mask_preds.sum() + mask_target.sum()
    score = (2 * intersection) / total
    
    return score


def testAccuracy(model, test_loader, threshold = 0.5):
    
    model.eval()
    score_list = []
    
    for batch in tqdm(test_loader):
        # Load Data to GPU
        images = batch["images"].cuda()
        shapes = batch["shapes"].long().cuda()
        # Forward Prop
        logits = model.forward(images)
        probs = logits.sigmoid()
        # get mIoU score (the cutting is done bc unet returns [b x c x h x w] with c = 2 (background = 0, object = 1))
        score = calculateAcc(probs.cpu().numpy()[:,1,...], shapes.cpu().numpy(), 1, threshold)
        # add to list
        score_list.append(score)
        
    return np.mean(score_list)


def calculateAcc(probs, target, class_id, threshold):
    
    mask_preds = probs > threshold
    mask_target = target == class_id
    
    correct = mask_preds == mask_target
    total = np.size(mask_preds)
    score = total / correct
    
    return score