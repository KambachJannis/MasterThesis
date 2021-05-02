import torch
import numpy as np
from tqdm.notebook import tqdm


def trainModel(model, optimizer, train_loader, criterion, mode):
    
    if mode == 'point':
        loss = trainPoint(model, optimizer, train_loader, criterion)
    elif mode == 'point_cob':
        loss = trainPointCOB(model, optimizer, train_loader, criterion)
    elif mode == 'supervised':
        loss = trainSupervised(model, optimizer, train_loader, criterion)
        
    return loss


@torch.no_grad()    
def valModel(model, val_loader, criterion, mode):
    
    if mode == 'point':
        loss = valPoint(model, val_loader, criterion)
    elif mode == 'point_cob':
        loss = valPointCOB(model, val_loader, criterion)
    elif mode == 'supervised':
        loss = valSupervised(model, val_loader, criterion)
        
    return loss


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
    
    for batch in tqdm(val_loader):
        # Load Data to GPU
        images = batch["images"].cuda()
        target = batch["points"].long().cuda()
        # Forward Prop
        logits = model.forward(images)
        probs = logits.sigmoid()
        # Calculate Loss
        loss = criterion(probs, target)
        loss_list.append(loss.item())
        
    return np.mean(loss_list)

    
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
    
    for batch in tqdm(val_loader):
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
        
    return np.mean(loss_list)

   
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
    
    for batch in tqdm(val_loader):
        # Load Data to GPU
        images = batch["images"].cuda()
        target = batch["shapes"].long().cuda()
        # Forward Prop
        logits = model.forward(images)
        probs = logits.sigmoid()
        # Calculate Loss
        loss = criterion(probs, target)
        loss_list.append(loss.item())
        
    return np.mean(loss_list)