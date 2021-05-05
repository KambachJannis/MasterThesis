import os
import torch
import numpy as np
from scipy import ndimage
import torch.nn.functional as F
from skimage.morphology import label as ski_label
from skimage.measure import regionprops as ski_regions
from skimage.segmentation import watershed as ski_watershed
from skimage.segmentation import find_boundaries as ski_boundaries

def computeLoss(probs, points, cob, roi_mask=None):
    """
    points: n x c x h x w
    probs: h x w (0 or 1)
    
    """
    loss = 0.
    # eliminate single dimensions (A x B x 1 -> A x B)
    points_red = points.squeeze()
    assert(points_red.max() <= 1)
    probs_red = probs.squeeze()
    cob_red = cob.squeeze()
    # unfold A x B tensor to A*B x 1 tensor
    probs_flat = probs_red.view(-1)
    
    # get list of all relevant pixels and their GT labels
    checklist = getPixelChecklist(points_red, probs_red, cob_red, roi_mask)
    
    for item in checklist:
        # get relevant pixels for this item
        item_ids = probs_flat[item['id_list']]
        if item['label'] is not None:
            # init tensor of same size and fill with label value
            item_label = torch.ones(item_ids.shape, device = item_ids.device) * item['label']
        else:
            item_label = item['labels'].float()
        # essentially log function
        loss += item['scale'] * F.binary_cross_entropy(item_ids, item_label, reduction='mean')
    
    return loss


@torch.no_grad()
def getPixelChecklist(points, probs, cob, roi_mask = None, batch = 0):
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
        id_background = pr_flat.argmin()
        checklist += [{'scale': 1, 'id_list': [id_background], 'label': 0, 'batch': batch}]

    if 1 in class_list:
        # pixel with highest chance for foreground class
        id_foreground = pr_flat.argmax()
        checklist += [{'scale': 1, 'id_list': [id_foreground], 'label': 1, 'batch': batch}]   

    ################ POINT LEVEL ######################
    if 1 in class_list:
        # locate point labels
        ids_labels = torch.where(pt_flat==1)[0]
        checklist += [{'scale': len(ids_labels), 'id_list': ids_labels, 'label': 1, 'batch': batch}]  

    ################ SPLIT LEVEL ######################
    probs_numpy = probs.detach().cpu().numpy()
    points_numpy = points.cpu().numpy()
    # get blobs
    blobs = getBlobs(probs_numpy, roi_mask=None)
    # get foreground and background blob ids
    foreground_blobs = np.unique(blobs * points_numpy) # mult with points = eliminate background
    background_blobs = [x for x in np.unique(blobs) if x not in foreground_blobs]

    if points_numpy.sum() > 1:
        # GLOBAL SPLIT
        boundaries = watersplit(probs_numpy, points_numpy)
        # get ids of boundary lines on flat array
        ids_boundaries = np.where(boundaries.ravel())[0]
        checklist += [{'scale': (points_numpy.sum()-1), 'id_list': ids_boundaries, 'label': 0, 'batch': batch}]  #boundaries = background = 0
        # LOCAL SPLIT
        for blob in foreground_blobs:
            # not background
            if blob == 0:
                continue
            # bool matrix with False where other blobs are, True for blob and background
            blob_mask = blobs==blob
            # all points within the mask (all points for this blob, since none on background)
            blob_points = points_numpy * blob_mask
            # skip blob if it does not need to be split
            if blob_points.sum() < 2: 
                continue
            # if not, initiate local split for only the selected blob points
            boundaries = watersplit(probs_numpy, blob_points) * blob_mask
            # get ids of boundary lines on flat array
            ids_boundaries = np.where(boundaries.ravel())[0]
            checklist += [{'scale': (blob_points.sum() - 1), 'id_list': ids_boundaries, 'label': 0, 'batch': batch}]  #boundaries = background = 0

    ################ FALSE POSITIVE LEVEL ######################
    for blob in background_blobs:
        # not background itself
        if blob == 0:
            continue
        # bool matrix with False where other blobs are, True for blob and background
        blob_mask = blobs==blob
        # subtract RoI mask if given
        if roi_mask is not None:
            blob_mask = (roi_mask * blob_mask)
        # ensure that blob has not been removed by RoI mask
        if blob_mask.sum() == 0:
            pass # do nothing
        else:
            # get ids of all the background pixels in RoI
            ids_background = np.where(blob_mask.ravel())[0]
            checklist += [{'scale': 1, 'id_list': ids_background, 'label': 0, 'batch': batch}]
            
    ################ CONVOLUTIONAL ORIENTED BOUNDARIES ######################
    
    for blob in foreground_blobs:
        if not os.path.isfile("blobs.pt"):
            torch.save(blobs, "blobs.pt")
            torch.save(cob, "cob.pt")
        # not background
        if blob == 0:
            continue
        # bool matrix with False where other blobs and background are, True for blob
        blob_mask = blobs==blob
        # pixel ids where blob is predicted
        ids_blob = np.where(blob_mask.ravel())[0]
        # apply mask to cob
        cob_preds = torch.round(cob[blob_mask])
        # check if objectness is white or black
        cob_preds_inv = (1 - cob_preds)
        if cob_preds_inv.sum() > cob_preds.sum():
            cob_preds = cob_preds_inv    
        
        checklist += [{'scale': 1, 'id_list': ids_blob, 'label': None, 'labels': cob_preds, 'batch': batch}]  #boundaries = background = 0


    return checklist


def watersplit(_probs, _points):
    '''
    Applies watershed and find_boundaries algorithms to detect object boundaries.
    Returns matrix with boundaries labeled 1.
    
    '''
    points = _points.copy()
    probs = _probs.copy()
    # assign 1 - n number to the n point labels
    points[points != 0] = np.arange(1, points.sum()+1)
    points = points.astype(float)
    # black top hat filter = remove all white objects that are too big
    # here: remove all blobs > 7
    probs = ndimage.black_tophat(probs, 7)
    # apply watershed algo to get image with segmented regions
    seg = ski_watershed(probs, points)
    # convert segmented regions into 0/1 boundaries
    splits = ski_boundaries(seg)

    return splits

def getBlobs(probs, roi_mask=None):
    ''' 
    Assignes unique numbers to connected high-probability regions.
    Essentially a transformation of the probability matrix.
    
    '''
    probs = probs.squeeze()
    h, w = probs.shape
    # zeros matrix with probs shape
    blobs = np.zeros((h, w), int)
    # discard unlikely regions
    pred_mask = (probs>0.5).astype('uint8')
    # connected pixel groups get assigned a unique number
    blobs = ski_label(pred_mask == 1)
    # subtract RoI mask if given
    if roi_mask is not None:
        blobs = (blobs * roi_mask[None]).astype(int)

    return blobs
        
def blobsToPoints(blobs):
    ''' 
    Exports map of blob centroids.
    
    '''
    blobs = blobs.squeeze()
    # init zeros array
    points = np.zeros(blobs.shape).astype("uint8")
    assert points.ndim == 2
    # returns a list of labeled regions with their properties
    region_properties = ski_regions(blobs)
    # iterate through regions (blobs)
    for blob in region_properties:
        # find central point
        y, x = blob.centroid
        # add point to empty array
        points[int(y), int(x)] = 1

    return points