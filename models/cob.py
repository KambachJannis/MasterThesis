import os
import tqdm
import torch
import numpy as np
from models import base
from helpers import metrics
from helpers import haven_viz
from PIL import Image, ImageDraw
from models.losses import cob_loss

class COB(torch.nn.Module):
    def __init__(self, exp_dict, n_classes):
        super().__init__()
        self.exp_dict = exp_dict
        self.n_classes = n_classes

        self.model_base = base.getBase(self.exp_dict['model']['base'], n_classes=self.n_classes)

        if self.exp_dict["optimizer"] == "adam":
            self.opt = torch.optim.Adam(self.model_base.parameters(), lr = self.exp_dict["lr"], betas = (0.99, 0.999), weight_decay = 0.0005)
        elif self.exp_dict["optimizer"] == "sgd":
            self.opt = torch.optim.SGD(self.model_base.parameters(), lr = self.exp_dict["lr"])
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
        # Declare Train Mode
        model.train()

        # Prepare Variables
        n_batches = len(train_loader)
        train_meter = metrics.Meter()
        pbar = tqdm.tqdm(total=n_batches)

        # MAIN LOOP
        for batch in train_loader:
            # Train on Batch
            score_dict = model.trainOnBatch(batch)
            # Save Loss
            train_meter.add(score_dict['train_loss'], batch['images'].shape[0])
            # Update PBar
            pbar.set_description("Training. Loss: %.4f" % train_meter.get_avg_score())
            pbar.update(1)

        overall_loss = train_meter.get_avg_score()
        pbar.close()

        return {'train_loss': overall_loss}
    
    def trainOnBatch(self, batch, **extras):
        # Zero the gradients 
        self.opt.zero_grad()
        # Declare Training Mode
        self.train()

        # Load Data to GPU
        images = batch["images"].cuda()
        points = batch["points"].long().cuda()
        cob = batch["cob"].cuda()
        # Forward Prop
        logits = self.model_base.forward(images)
        probs = logits.sigmoid()
        # Calculate Loss
        loss = cob_loss.computeLoss(points = points, probs = probs, cob = cob)
        # Backprop
        loss.backward()
        # Optimize
        self.opt.step()

        return {"train_loss": loss.item()}

    @torch.no_grad()
    def valOnLoader(self, val_loader, savedir_images = None, n_images = 2):
        # Declare Eval Mode
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
        # Declare Eval Mode
        self.eval()

        # Load Data to GPU
        images = batch["images"].cuda()
        points = batch["points"].long().cuda()
        
        # Forward Prop
        logits = self.model_base.forward(images)
        # Sigmoid Function
        probs = logits.sigmoid().cpu().numpy()
        blobs = cob_loss.getBlobs(probs=probs)
        miscounts = abs(float((np.unique(blobs) !=0 ).sum() - (points != 0).sum()))

        return {'miscounts': miscounts}
        
    @torch.no_grad()
    def visOnBatch(self, batch, savedir_image, text_in = None):
        # Declare Eval Mode
        self.eval()
        
        ############ Preprocess ########################
        # Get Image for Prediction
        images = batch["images"].cuda()
        # Get Probs for Image
        logits = self.model_base.forward(images)
        probs = logits.sigmoid().cpu().numpy()
        # Unique Labels for each blob
        blobs = cob_loss.getBlobs(probs = probs)
        # Remove single Dimensions
        pred_blobs = blobs.squeeze()
        pred_probs = probs.squeeze()
        # Get Denormalized Image to Display
        img_src = haven_viz.get_image(batch["images"], denorm="rgb")

        ################ GROUND TRUTH POINTS ######################
        points = batch["points"][0].long().numpy().squeeze()
        n_points = batch["points"].sum().item()
        y_list, x_list = np.where(points)
        
        if text_in is not None:
            text = text_in
        else:
            text = f"{n_points}"
        
        img_labels = haven_viz.points_on_image(y_list, x_list, img_src)
        
        shapes = batch["shapes"]
        if len(shapes) > 0:
            img = Image.new('L', (250, 250), 0)
            for shape in shapes:
                flat_list = [item.item() for sublist in shape for item in sublist]
                ImageDraw.Draw(img).polygon(flat_list, outline=1, fill=1) 
            
            img_labels = haven_viz.mask_on_image(img_labels, np.array(img).squeeze())
        
        haven_viz.text_on_image(text=text, image=img_labels)

        ################### BLOBS ################################
        pred_points = cob_loss.blobsToPoints(pred_blobs).squeeze()
        y_list, x_list = np.where(pred_points.squeeze())
        text = f"{len(y_list)}"
        
        img_pred = haven_viz.mask_on_image(img_src, pred_blobs)
        # img_pred = haven_img.points_on_image(y_list, x_list, img_org) in case I want points inside the blobs
        
        haven_viz.text_on_image(text=text, image=img_pred)

        #################### HEATMAP #############################
        heatmap = haven_viz.gray2cmap(pred_probs)
        heatmap = haven_viz.f2l(heatmap)
        #haven_viz.text_on_image(text="lcfcn heatmap", image=heatmap)
    
        ################# Stitch and Save #########################
        stitched_image = np.hstack([img_labels, img_pred, heatmap])
        haven_viz.save_image(savedir_image, stitched_image)
        
    @torch.no_grad()
    def testOnLoader(self, test_loader, savedir_images = None, n_images = 2):
        # Declare Eval Mode
        self.eval()
        
        # Prepare Variables
        n_batches = len(test_loader)
        test_meter = metrics.Meter()
        pbar = tqdm.tqdm(total=n_batches)
        
        # MAIN LOOP
        for i, batch in enumerate(test_loader):
            # Validate on Batch
            score_dict = self.testOnBatch(batch)
            # Save Score
            test_meter.add(score_dict['mIoU'], batch['images'].shape[0])
            # Update PBar
            pbar.set_description("Testing. mIoU: %.4f" % test_meter.get_avg_score())
            pbar.update(1)
            # Export Demo Images
            if savedir_images and i < n_images:
                os.makedirs(savedir_images, exist_ok = True)
                self.visOnBatch(batch, savedir_image=os.path.join(savedir_images, "%d_test.jpg" % i), text_in = str(score_dict['mIoU']))
                
        pbar.close()
        test_mIoU = test_meter.get_avg_score()
        test_dict = {'test_mIoU': test_mIoU}
        
        return test_dict

    def testOnBatch(self, batch):
        # Declare Eval Mode
        self.eval()

        # Load Data to GPU
        images = batch["images"].cuda()
        points = batch["points"].long().cuda()
        shapes = batch["shapes"]
        points_numpy = points.cpu().numpy()
        
        # Forward Prop
        logits = self.model_base.forward(images)
        # Sigmoid Function
        probs = logits.sigmoid().cpu().numpy()
        blobs = cob_loss.getBlobs(probs=probs)
        
        # iterate over shapes, to ensure that GOOD predictions without label are not penalized
        running_iou = 0
        for shape in shapes:
            # flat list of coordinates [x1, y1, x2, y2, ...]
            flat_list = [item.item() for sublist in shape for item in sublist]
            # draw template image
            img = Image.new('L', (250, 250), 0)
            # draw filled polygon into template
            ImageDraw.Draw(img).polygon(flat_list, outline=1, fill=1)
            # convert to bool-mask
            shape_numpy = np.array(img)
            shape_mask = shape_numpy == 1
            
            # find out ids of blobs that intersect with shape
            blob_ids = np.unique(blobs * shape_mask)
            blob_ids = blob_ids[blob_ids != 0] # not background
            
            # compute mIoU with logical numpy ops
            if len(blob_ids) == 0:
                running_iou += 0
            else:
                # collect bool masks of all blobs
                blob_masks = []
                for blob in blob_ids:
                    blob_mask = blobs == blob
                    intersection = np.logical_and(blob_mask, shape_mask).sum()
                    if intersection > (0.5 * blob_mask.sum()): 
                        blob_masks.append(blob_mask)
                
                # all blobs might be eliminated, 1 and more need to be treated seperately
                if len(blob_masks) == 0:
                    running_iou += 0
                elif len(blob_masks) == 1:
                    blob_union = blob_masks[0]
                    union = np.logical_or(blob_union, shape_mask).sum()
                    intersection = np.logical_and(blob_union, shape_mask).sum()
                    running_iou += intersection / union
                else:
                    # combine into single blob mask
                    blob_union = np.logical_or.reduce(blob_masks)
                    union = np.logical_or(blob_union, shape_mask).sum()
                    intersection = np.logical_and(blob_union, shape_mask).sum()
                    running_iou += intersection / union
                
        return {'mIoU': running_iou / len(shapes)}