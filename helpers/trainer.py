import torch
import numpy as np
from tqdm.notebook import tqdm

def trainModel(model, optimizer, train_loader, criterion):
    
    model.train()
    loss_list = [] 
    
    for batch in tqdm(train_loader):
        # Zero Gradients
        optimizer.zero_grad()
        # Load Data to GPU
        images = batch["images"].cuda()
        if 'shapes' in batch:
            target = batch["shapes"].long().unsqueeze(0).cuda()
        else:
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


@torch.no_grad()    
def valModel(model, val_loader, criterion):
    
    model.eval()
    loss_list = [] 
    
    for batch in tqdm(val_loader):
        # Load Data to GPU
        images = batch["images"].cuda()
        points = batch["points"].long().cuda()
        # Forward Prop
        logits = model.forward(images)
        probs = logits.sigmoid()
        # Calculate Loss
        loss = criterion(probs, points)
        loss_list.append(loss.item())
        
    return np.mean(loss_list)