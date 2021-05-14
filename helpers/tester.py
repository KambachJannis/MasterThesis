import cv2
import torch
import torchvision
import numpy as np
import matplotlib as plt
from skimage.io import imread
from skimage.segmentation import mark_boundaries
from skimage.morphology import label as ski_label
from tqdm.notebook import tqdm


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
    if intersection == 0 or union == 0:
        score = 0
    else:
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
    if intersection == 0 or total == 0:
        score = 0
    else:
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
    if correct == 0 or total == 0:
        score = 0
    else:
        score = correct / total 
    
    return score


@torch.no_grad()
def imagesToTB(model, test_loader, TB, amount_batches = 30, threshold = 0.5):
    
    model.eval()
    counter = amount_batches
    
    for batch in tqdm(test_loader):
        if counter > 0:
            # Load Data to GPU
            images = batch["images"].cuda()
            #points = batch["points"].long().numpy()
            shapes = batch["shapes"].long().numpy()
            paths = batch['meta']['path']
            # Forward Prop
            logits = model.forward(images)
            probs = logits.sigmoid()
            # convert
            probs = probs.cpu().detach().numpy()[:,-1,...]
            # iterate over images
            for i in range(len(probs)):
                image_src = imread(paths[i])
                # calculate scores for image
                mIoU = calculateMIoU(probs[i], shapes[i], 1, threshold) * 100
                dice = calculateDice(probs[i], shapes[i], 1, threshold) * 100
                pAcc = calculatepAcc(probs[i], shapes[i], 1, threshold) * 100
                label = f"IoU: {np.round(mIoU, 2)}%, Dice: {np.round(dice, 2)}, Pixel Accuracy: {np.round(pAcc, 2)}%"
                # draw ground truth image
                labels = drawShapes(image_src, shapes[i])
                #labels = drawPoints(labels, points[i])
                # draw predicted blobs
                blobs = ski_label((probs[i] > threshold).astype('uint8') == 1)
                preds = drawShapes(image_src, blobs)
                # draw heatmap
                heatmap = drawHeatmap(probs[i])
                # stack and add to tensorboard
                images = np.array([labels, preds, heatmap])
                images = np.moveaxis(images, -1, 1) #RGB Dimension (last, -1) to second dimension
                img_grid = torchvision.utils.make_grid(torch.from_numpy(images))
                TB.add_image(label, img_grid)
        
            counter -= 1
            
# NOTE: color is BGR           
def drawPoints(image, points, color = (0, 0, 255), thickness = 2):
    
    y_list, x_list = np.where(points)
    h, w, _ = image.shape
    img = image.copy()
    
    for i, (y, x) in enumerate(zip(y_list, x_list)):
        if y < 1:
            x, y = int(x*W), int(y*H) 
        else:
            x, y = int(x), int(y) 
            
        result = cv2.circle(img, (x,y), 2, color, thickness) 

    return result
    

def drawShapes(image, shapes):
    
    objects = np.unique(shapes)
    red = np.zeros(image.shape, dtype='uint8')
    red[:,:,2] = 255
    alpha = 0.5
    result = image.copy()
    
    for obj in objects:
        if obj == 0: continue
        mask = shapes == obj
        result[mask] = result[mask] * alpha + red[mask] * (1 - alpha)
        
    result = mark_boundaries(result, shapes) 

    return result


def drawHeatmap(image):
    
    img = image.copy()
    img = img / max(1, img.max()) #scale to max as 1
    img = np.maximum(img, 0) # zero negative probs
    img = img / max(1, img.max()) # no changes
    img = img * 255 # scale to 0-255

    img = img.astype(int)
    cmap = plt.cm.get_cmap("jet")
    result = np.zeros(img.shape + (3, ), dtype=np.float64)

    for c in np.unique(img):
        result[(img == c).nonzero()] = cmap(c)[:3]
        
    return result