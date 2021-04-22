import os
import tqdm
import numpy as np
from fnmatch import fnmatch
from collections import defaultdict

import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader

from models.base import cobnet
from helpers.cob.dataset import PASCALDataSet
        
def initRetrain(cfg):
        if (not os.path.exists(cfg.run)):
            os.makedirs(cfg.run)

        # init model
        model = cobnet.CobNet()
        device = torch.device('cuda')
        model.to(device)
        model_params = parseModelParams(model)

        # init PASCAL data loader
        train_set = PASCALDataSet(imgs = cfg.images, segs = cfg.segments, split = 'train')
        train_loader_fs = DataLoader(train_set, collate_fn = train_set.collate_fn, batch_size = 16, drop_last = True, shuffle = True)
        train_loader_or = DataLoader(train_set, collate_fn = train_set.collate_fn, batch_size = 4, drop_last = True, shuffle = True)

        val_set = PASCALDataSet(imgs = cfg.images, segs = cfg.segments, split = 'val')
        val_loader = DataLoader(val_set, collate_fn = val_set.collate_fn, batch_size = 32)
        prev_loader = DataLoader(val_set, collate_fn = val_set.collate_fn, batch_size = 10)

        dataloaders = {
            'train_fs': train_loader_fs,
            'train_or': train_loader_or,
            'prev': prev_loader,
            'val': val_loader
        }

        optimizers = {'base': optim.SGD([{'params': model_params['base0-3.weight'],'lr': cfg.lr},
                                 {'params': model_params['base0-3.bias'], 'weight_decay': 0, 'lr': cfg.lr * 2},
                                 {'params': model_params['base4.weight'], 'lr': cfg.lr * 100},
                                 {'params': model_params['base4.bias'], 'weight_decay': 0, 'lr': cfg.lr * 200},],
                                lr=cfg.lr, weight_decay=cfg.decay, momentum=cfg.momentum),
              'reduc': optim.SGD([{'params': model_params['reducers.weight'], 'lr': cfg.lr * 100,}, 
                                  {'params': model_params['reducers.bias'], 'lr': cfg.lr * 200, 'weight_decay': 0,}],
                                 weight_decay=cfg.decay, momentum=cfg.momentum),
              'fuse': optim.SGD([{'params': model_params['fuse.weight'], 'lr': cfg.lr * 100,}, 
                                 {'params': model_params['fuse.bias'], 'lr': cfg.lr * 200, 'weight_decay': 0,}],
                                weight_decay=cfg.decay, momentum=cfg.momentum),
              'orientation': optim.SGD([{'params': model_params['orientation.weight'], 'lr': cfg.lr}, 
                                        {'params': model_params['orientation.bias'], 'lr': cfg.lr * 2, 'weight_decay': 0,}],
                                       weight_decay=cfg.decay, momentum=cfg.momentum),
        }

        lr_scheduler = {'base': optim.lr_scheduler.MultiStepLR(optimizers['base'], milestones=[cfg.epochs_div_lr], gamma=0.1),
                        'reduc': optim.lr_scheduler.MultiStepLR(optimizers['reduc'], milestones=[cfg.epochs_div_lr], gamma=0.1),
                        'fuse': optim.lr_scheduler.MultiStepLR(optimizers['fuse'], milestones=[cfg.epochs_div_lr], gamma=0.1)
        }

        if (os.path.exists(os.path.join(cfg.run, 'checkpoints', 'cp_fs.pth.tar'))):
            print('found model, will skip fusion mode')
            start_epoch = 8
            mode = 'or'
        else:
            print('creating new model')
            start_epoch = 0
            mode = 'fs'

        for epoch in range(start_epoch, cfg.epochs):
            if (epoch > 7):
                mode = 'or'

            print('epoch {}/{}, mode: {}, lr: {:.2e}'.format(epoch + 1, 10, mode, lr_scheduler['base'].get_last_lr()[0]))

            model.train()
            model.base_model.apply(freeze)

            losses = trainOneEpoch(model, dataloaders, optimizers, mode, epoch)

            for k in lr_scheduler.keys():
                lr_scheduler[k].step()

            # save checkpoint
            save_path = os.path.join(cfg.run, 'checkpoints', 'cp_{}.pth.tar'.format(mode))
            state_dict = model.state_dict()
            torch.save(state_dict, save_path)
            
            
def trainOneEpoch(model, dataloaders, optimizers, mode, epoch):

    running_loss = 0
    criterion = BalancedBCE()

    if (mode == 'fs'):
        dataloader = dataloaders['train_fs']
    else:
        dataloader = dataloaders['train_or']

    pbar = tqdm(total=len(dataloader))
    for i, data in enumerate(dataloader):
        
        data = batchToDevice(data, device)
        loss = 0
        
        with torch.set_grad_enabled(True):
            
            if (mode == 'fs'):
                
                # Forward Sides
                _, sides = model.forward_sides(data['image'])
                for s in sides:
                    loss += criterion(s, data['cntr'])
                
                # Forward Fuse
                y_fine, y_coarse = model.forward_fuse(sides)
                loss += criterion(y_fine, data['cntr'])
                loss += criterion(y_coarse, data['cntr'])
                
                # Backprop
                loss.backward()
                
                # Opt and Zero Grad
                for opt in optimizers.keys():
                    optimizers[opt].step()
                    optimizers[opt].zero_grad()

                # Sum
                running_loss += loss.cpu().detach().numpy()

            else:
                
                # Zero Grads
                optimizers['orientation'].zero_grad()
                
                #Forward
                res = model(data['image'])
                for i, _ in enumerate(res['orientations']):
                    loss += criterion(res['orientations'][i], (data['or_cntr'] == i + 1).float())
                
                #Backprop
                loss.backward()
                
                # Opt
                optimizers['orientation'].step()
                
                # Sum 
                running_loss += loss.cpu().detach().numpy()

        loss = running_loss / ((i + 1) * dataloader.batch_size)
        pbar.set_description('[train] loss {:.3e}'.format(loss))
        pbar.update(1)

    pbar.close()
    loss = running_loss / (dataloader.batch_size * len(dataloader))
    out = {'train/loss_{}'.format(mode): loss}

    return out
            
            
def parseModelParams(model):
    skipped_names = []
    added_names = []
    model_params = defaultdict(list)
    
    for name, param in model.named_parameters():
        if 'base_model' in name:
            if (fnmatch(name, '*layer[123]*') and 'conv' in name) or 'conv' in name:
                if 'weight' in name:
                    model_params['base0-3.weight'].append(param)
                else:
                    model_params['base0-3.bias'].append(param)
            else:
                if 'weight' in name:
                    model_params['base4.weight'].append(param)
                else:
                    model_params['base4.bias'].append(param)
        elif 'reducer' in name:
            if 'weight' in name:
                model_params['reducers.weight'].append(param)
            else:
                model_params['reducers.bias'].append(param)
        elif 'fuse' in name:
            if 'weight' in name:
                model_params['fuse.weight'].append(param)
            else:
                model_params['fuse.bias'].append(param)
        elif 'orientation' in name:
            if 'weight' in name:
                model_params['orientation.weight'].append(param)
            else:
                model_params['orientation.bias'].append(param)

    return model_params

def freeze(m):
    # we update the running stats and freeze gamma/beta only
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        pass
    
def batchToDevice(batch, device):
    return {
        k: v.to(device) if (isinstance(v, torch.Tensor)) else v
        for k, v in batch.items()
    }

class BalancedBCE(nn.Module):
    
    def __init__(self):
        super(BalancedBCE, self).__init__()

    def forward(self, input, target):
        bce = F.binary_cross_entropy_with_logits(input, target, reduction='none')

        bsize = target.shape[0]
        beta_pos = torch.tensor([1 - t.sum() / t.numel() for t in target])
        beta_neg = torch.tensor([t.sum() / t.numel() for t in target])
        loss_pos = torch.cat([beta_pos[i] * bce[i][target[i] == 1] for i in range(bsize)]).mean()
        loss_neg = torch.cat([beta_neg[i] * bce[i][target[i] == 0] for i in range(bsize)]).mean()

        return loss_pos + loss_neg
