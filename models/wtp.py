import os
import tqdm
import torch
import numpy as np
from models import base
from helpers import metrics
from helpers import haven_viz
from models.losses import wtp_loss
import matlab.engine

class WTP(torch.nn.Module):
    def __init__(self, exp_dict, n_classes):
        super().__init__()
        self.exp_dict = exp_dict
        self.n_classes = n_classes

        self.model_base = base.getBase(self.exp_dict['model']['base'], self.exp_dict, n_classes=self.n_classes)

        if self.exp_dict["optimizer"] == "adam":
            self.opt = torch.optim.Adam(self.model_base.parameters(), lr = self.exp_dict["lr"], betas = (0.99, 0.999), weight_decay = 0.0005)
        elif self.exp_dict["optimizer"] == "sgd":
            self.opt = torch.optim.SGD(self.model_base.parameters(), lr = self.exp_dict["lr"], momentum = 0.9, weight_decay = 0.0005)
        else:
            name = self.exp_dict["optimizer"]
            raise ValueError(f"Optimizer {name} not integrated.")

    def getStateDict(self):
        state_dict = {"model": self.model_base.state_dict(), "opt":self.opt.state_dict()}
        return state_dict

    def loadStateDict(self, state_dict):
        self.model_base.load_state_dict(state_dict["model"])
        self.opt.load_state_dict(state_dict["opt"])
    
    def trainOnLoader(self, model, train_loader):
        # ???
        model.train()
        # start matlab engine
        eng = matlab.engine.start_matlab()
        eng.addpath('/home/jovyan/work/ma/helpers/objectness', nargout=0)
        eng.addpath('/home/jovyan/work/ma/helpers/objectness/pff_segment', nargout=0)
        eng.addpath('/home/jovyan/work/ma/helpers/objectness/MEX', nargout=0)
        # Prepare Variables
        n_batches = len(train_loader)
        train_meter = metrics.Meter()
        pbar = tqdm.tqdm(total=n_batches)

        # MAIN LOOP
        for batch in train_loader:
            # Train on Batch
            score_dict = model.trainOnBatch(batch, eng)
            # Save Loss
            train_meter.add(score_dict['train_loss'], batch['images'].shape[0])
            # Update PBar
            pbar.set_description("Training. Loss: %.4f" % train_meter.get_avg_score())
            pbar.update(1)

        overall_loss = train_meter.get_avg_score()
        pbar.close()
        eng.quit()

        return {'train_loss': overall_loss}
    
    def trainOnBatch(self, batch, eng, **extras):
        # Zero the gradients 
        self.opt.zero_grad()
        # ???
        self.train()

        # Load Data to GPU
        images = batch["images"].cuda()
        points = batch["points"].long().cuda()
        
        # Forward Prop
        logits = self.model_base.forward(images)
        # Calculate Loss
        loss = wtp_loss.computeLoss(points = points, probs = logits.sigmoid(), images = batch["meta"]["path"][0], eng = eng)
        # Backprop
        loss.backward()
        # Optimize
        self.opt.step()

        return {"train_loss": loss.item()}

    @torch.no_grad()
    def valOnLoader(self, val_loader, savedir_images = None, n_images = 2):
        # ???
        self.eval()
        
        # Prepare Variables
        n_batches = len(val_loader)
        val_meter = metrics.Meter()
        pbar = tqdm.tqdm(total=n_batches)
        
        # MAIN LOOP
        for i, batch in enumerate(val_loader):
            # Validate on Batch
            score_dict = self.valOnBatch(batch)
            # Save Score
            val_meter.add(score_dict['miscounts'], batch['images'].shape[0])
            # Update PBar
            pbar.set_description("Validating. MAE: %.4f" % val_meter.get_avg_score())
            pbar.update(1)
            # Export Demo Images
            if savedir_images and i < n_images:
                os.makedirs(savedir_images, exist_ok = True)
                self.visOnBatch(batch, savedir_image=os.path.join(savedir_images, "%d.jpg" % i))
                
        pbar.close()
        val_mae = val_meter.get_avg_score()
        val_dict = {'val_mae':val_mae, 'val_score':-val_mae}
        
        return val_dict

    def valOnBatch(self, batch):
        #???
        self.eval()

        # Load Data to GPU
        images = batch["images"].cuda()
        points = batch["points"].long().cuda()
        
        # Forward Prop
        logits = self.model_base.forward(images)
        # ??
        probs = logits.sigmoid().cpu().numpy()
        blobs = wtp_loss.getBlobs(probs=probs)
        miscounts = abs(float((np.unique(blobs) !=0 ).sum() - (points != 0).sum()))

        return {'miscounts': miscounts}
        
    @torch.no_grad()
    def visOnBatch(self, batch, savedir_image):
        self.eval()
        images = batch["images"].cuda()
        #points = batch["points"].long().cuda() unused var
        logits = self.model_base.forward(images)
        probs = logits.sigmoid().cpu().numpy()

        blobs = wtp_loss.getBlobs(probs=probs)

        #pred_counts = (np.unique(blobs)!=0).sum() unused var
        pred_blobs = blobs
        pred_probs = probs.squeeze()

        # loc 
        #pred_count = pred_counts.ravel()[0] #unused var
        pred_blobs = pred_blobs.squeeze()
        
        img_org = haven_viz.get_image(batch["images"],denorm="rgb")

        # true points
        y_list, x_list = np.where(batch["points"][0].long().numpy().squeeze())
        img_peaks = haven_viz.points_on_image(y_list, x_list, img_org)
        text = "%s ground truth" % (batch["points"].sum().item())
        haven_viz.text_on_image(text=text, image=img_peaks)

        # pred points 
        pred_points = wtp_loss.blobsToPoints(pred_blobs).squeeze()
        y_list, x_list = np.where(pred_points.squeeze())
        img_pred = haven_viz.mask_on_image(img_org, pred_blobs)
        # img_pred = haven_img.points_on_image(y_list, x_list, img_org)
        text = "%s predicted" % (len(y_list))
        haven_viz.text_on_image(text=text, image=img_pred)

        # heatmap 
        heatmap = haven_viz.gray2cmap(pred_probs)
        heatmap = haven_viz.f2l(heatmap)
        haven_viz.text_on_image(text="lcfcn heatmap", image=heatmap)
    
        img_mask = np.hstack([img_peaks, img_pred, heatmap])
        
        haven_viz.save_image(savedir_image, img_mask)
     