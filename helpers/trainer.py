import torch
import numpy as np
from tqdm.notebook import tqdm


def trainModel(model, optimizer, train_loader, criterion, mode):
    
    if mode == 'point':
        loss = trainPoint(model, optimizer, train_loader, criterion)
    elif mode == 'point_cob':
        loss = trainPointCOB(model, optimizer, train_loader, criterion)
    elif mode == 'mixed':
        loss = trainMixed(model, optimizer, train_loader, criterion)
    elif mode == 'supervised':
        loss = trainSupervised(model, optimizer, train_loader, criterion)
    else:
        raise ValueError("wrong train mode given")
        
    return loss


@torch.no_grad()    
def valModel(model, val_loader, criterion, mode):
    
    if mode == 'point':
        loss_dict = valPoint(model, val_loader, criterion)
    elif mode == 'point_cob':
        loss = valPointCOB(model, val_loader, criterion)
    elif mode == 'mixed':
        loss = valPoint(model, val_loader, criterion) # or valPointCOB, idea is to no used full labels in val set
    elif mode == 'supervised':
        loss = valSupervised(model, val_loader, criterion)
    else:
        raise ValueError("wrong val mode given")
        
    return loss_dict


def trainPoint(model, optimizer, train_loader, criterion):
    
    model.train()
    loss_list = [] 
    
    for batch in tqdm(train_loader):
        # Zero Gradients
        optimizer.zero_grad()
        # Load Data to GPU
        images = batch["images"].cuda()
        target = batch["points"].long().cuda()
        # Forward Prop
        logits = model.forward(images)
        probs = logits.sigmoid()
        # Calculate Loss
        loss = criterion(probs, target)
        loss_list.append(loss.item())
        # Backprop
        loss.backward()
        # Step
        optimizer.step()
        
    return np.mean(loss_list)


def valPoint(model, val_loader, criterion):
             
    model.eval()
    loss_list = []
    mIoU_list = []
    
    for batch in tqdm(val_loader):
        # Load Data to GPU
        images = batch["images"].cuda()
        # check labels
        if batch["label_p"] == 0 or batch["label_s"] == 0:
            raise ValueError("A sample in the validation set is missing either point- or shape-labels.") 
        # more loading
        target = batch["points"].long().cuda()
        shapes = batch["shapes"].long().cuda()
        # Forward Prop
        logits = model.forward(images)
        probs = logits.sigmoid()
        # Calculate Loss
        loss = criterion(probs, target)
        loss_list.append(loss.item())
        # get mIoU score (the cutting is done bc unet returns [b x c x h x w] with c = 2 (background = 0, object = 1)) 
        probs = probs.cpu().detach().numpy()
        mIoU = calculateMIoU(probs[:,-1,...], shapes.cpu().numpy(), 1, 0.5)
        mIoU_list.append(mIoU)
        
    return {'loss': np.mean(loss_list),
            'mIoU': np.mean(mIoU_list)}

    
def trainPointCOB(model, optimizer, train_loader, criterion):
    
    model.train()
    loss_list = [] 
    
    for batch in tqdm(train_loader):
        # Zero Gradients
        optimizer.zero_grad()
        # Load Data to GPU
        images = batch["images"].cuda()
        target = batch["points"].long().cuda()
        cob = batch["cob"].cuda()
        # Forward Prop
        logits = model.forward(images)
        probs = logits.sigmoid()
        # Calculate Loss
        loss = criterion(probs, target, cob)
        loss_list.append(loss.item())
        # Backprop
        loss.backward()
        # Step
        optimizer.step()
        
    return np.mean(loss_list)


def valPointCOB(model, val_loader, criterion):
             
    model.eval()
    loss_list = []
    mIoU_list = []
    
    for batch in tqdm(val_loader):
        # Load Data to GPU
        images = batch["images"].cuda()
        # check labels
        if batch["label_p"] == 0 or batch["label_s"] == 0 or batch["label_c"] == 0:
            raise ValueError("A sample in the validation set is missing either point-, cob-, or shape-labels.") 
        # more loading
        target = batch["points"].long().cuda()
        shapes = batch["shapes"].long().cuda()
        cob = batch["cob"].cuda()
        # Forward Prop
        logits = model.forward(images)
        probs = logits.sigmoid()
        # Calculate Loss
        loss = criterion(probs, target, cob)
        loss_list.append(loss.item())
        # get mIoU score (the cutting is done bc unet returns [b x c x h x w] with c = 2 (background = 0, object = 1)) 
        probs = probs.cpu().detach().numpy()
        mIoU = calculateMIoU(probs[:,-1,...], shapes.cpu().numpy(), 1, 0.5)
        mIoU_list.append(mIoU)
        
    return {'loss': np.mean(loss_list),
            'mIoU': np.mean(mIoU_list)}

   
def trainSupervised(model, optimizer, train_loader, criterion):
    
    model.train()
    loss_list = [] 
    
    for batch in tqdm(train_loader):
        # Zero Gradients
        optimizer.zero_grad()
        # Load Data to GPU
        images = batch["images"].cuda()
        target = batch["shapes"].long().cuda()
        # Forward Prop
        logits = model.forward(images)
        probs = logits.sigmoid()
        # Calculate Loss
        loss = criterion(probs, target)
        loss_list.append(loss.item())
        # Backprop
        loss.backward()
        # Step
        optimizer.step()
        
    return np.mean(loss_list)


def valSupervised(model, val_loader, criterion):
             
    model.eval()
    loss_list = []
    mIoU_list = []
    
    for batch in tqdm(val_loader):
        # Load Data to GPU
        images = batch["images"].cuda()
        if batch["label_s"] == 0:
            raise ValueError("A sample in the validation set is missing shape-labels.") 
        # more loading
        target = batch["shapes"].long().cuda()
        # Forward Prop
        logits = model.forward(images)
        probs = logits.sigmoid()
        # Calculate Loss
        loss = criterion(probs, target)
        loss_list.append(loss.item())
        # get mIoU score (the cutting is done bc unet returns [b x c x h x w] with c = 2 (background = 0, object = 1)) 
        probs = probs.cpu().detach().numpy()
        mIoU = calculateMIoU(probs[:,-1,...], shapes.cpu().numpy(), 1, 0.5)
        mIoU_list.append(mIoU)
        
    return {'loss': np.mean(loss_list),
            'mIoU': np.mean(mIoU_list)}


def trainMixed(model, optimizer, train_loader, criterion):
    
    model.train()
    loss_list = [] 
    
    for batch in tqdm(train_loader):
        # Zero Gradients
        optimizer.zero_grad()
        # Load Data to GPU
        images = batch["images"].cuda()
        # Forward Prop
        logits = model.forward(images)
        probs = logits.sigmoid()
        
        # Load Target --------------------------------might need to load COB here too
        if batch['label_s'] == 1:
            target = batch["shapes"].long().cuda()
            criterion2 = torch.nn.CrossEntropyLoss()
            loss = criterion2(probs, target)
        elif batch['labels_p'] == 1:
            target = batch["points"].long().cuda()
            loss = criterion(probs, target)
        else:
            raise ValueError("dataset wrong, no label found")
        
        # Calculate Loss
        loss_list.append(loss.item())
        # Backprop
        loss.backward()
        # Step
        optimizer.step()
        
    return np.mean(loss_list)

def calculateMIoU(probs, target, class_id, threshold = 0.5):
    
    mask_preds = probs > threshold
    mask_target = target == class_id
    
    intersection = np.logical_and(mask_preds, mask_target).sum()
    union = np.logical_or(mask_preds, mask_target).sum()
    if intersection == 0 or union == 0:
        score = 0
    else:
        score = intersection / union
    
    return score
        