import torch
import numpy as np
from tqdm.notebook import tqdm
from skimage.io import imread

@torch.no_grad()
def testModel(model, test_loader, metric_list):
    
    results_dict = {}
    
    if 'mIoU' in metric_list:
        results_dict['mIoU'] = testMIoU(model, test_loader)
    if 'dice' in metric_list:
        results_dict['dice'] = testDice(model, test_loader)
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
        probs = probs.cpu().detach().numpy()
        # get mIoU score (the cutting is done bc unet returns [b x c x h x w] with c = 2 (background = 0, object = 1)) 
        score = calculateMIoU(probs[:,-1,...], shapes.cpu().numpy(), 1, threshold)
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
        probs = probs.cpu().detach().numpy()
        # get mIoU score (the cutting is done bc unet returns [b x c x h x w] with c = 2 (background = 0, object = 1))
        score = calculateDice(probs[:,-1,...], shapes.cpu().numpy(), 1, threshold)
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
        probs = probs.cpu().detach().numpy()
        # get mIoU score (the cutting is done bc unet returns [b x c x h x w] with c = 2 (background = 0, object = 1))
        score = calculatepAcc(probs[:,-1,...], shapes.cpu().numpy(), 1, threshold)
        # add to list
        score_list.append(score)
        
    return np.mean(score_list)


def calculatepAcc(probs, target, class_id, threshold):
    
    mask_preds = probs > threshold
    mask_target = target == class_id
    
    correct = (mask_preds == mask_target).sum()
    total = np.size(mask_preds)
    score = correct / total
    
    return score


@torch.no_grad()
def imagesToTB(model, test_loader, TB, amount_batches = 30, threshold = 0.5):
    
    model.eval()
    counter = amount_batcher
    
    for batch in tqdm(test_loader):
        if counter > 0:
            # Load Data to GPU
            images = batch["images"].cuda()
            points = batch["points"].long().cuda()
            shapes = batch["shapes"].long().cuda()
            paths = batch['meta']['path']
            # Forward Prop
            logits = model.forward(images)
            probs = logits.sigmoid()
            # convert
            probs = probs.cpu().detach().numpy()[:,-1,...]
            shapes = shapes.cpu().numpy()
            points = points.cpu().numpy()
            
            for i in range(len(probs)):
                prob = probs[i]
                point = points[i]
                shape = shapes[i]
                
                mIoU = calculateMIoU(prob, shape, 1, threshold)
                dice = calculateDice(prob, shape, 1, threshold)
                pAcc = calculatepAcc(prob, shape, 1, threshold)
                label = f"IoU: {mIoU}, Dice: {dice}, Pixel Accuracy: {pAcc}"
                
                image_src = imread(paths[i])
                labels = drawLabels(image_src, point, shape)
                preds = drawPreds(image_src, prob, threshold)
                heatmap = drawHeatmap(image_src, prob)
            
                img_grid = torchvision.utils.make_grid(images)
                TB.add_image(label, img_grid)
        
            counter -= 1
            
            
def drawLabels(image, points, shapes):
    pass

def drawPreds(image, preds, threshold):
    pass

def drawHeatmap(image, preds):
    pass