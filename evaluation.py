import numpy as np
import torch
from tqdm import tqdm
import wandb
import numpy as np
import torch.nn.functional as F
def mpa(preds, masks):

    #Number of correct pixels predicted
    preds = preds.argmax(dim=1).reshape(-1)
    masks = masks.argmax(dim=1).reshape(-1)

    correct_pixels = torch.sum(preds == masks)
    total_pixels = len(preds)
    pixel_acc = correct_pixels / total_pixels

    return pixel_acc

def iou(preds, masks):
    #Ratio between AoI and AoU (between the predicted mask and ground truth mask of a single class)
    flat_preds = preds.argmax(dim=1).reshape(-1)
    flat_masks = masks.argmax(dim=1).reshape(-1)
    intersection = torch.sum(flat_preds == flat_masks)

    gt_pixels = torch.sum(torch.ones_like(masks.argmax(dim=1)))
    pred_pixels = torch.sum(torch.ones_like(preds.argmax(dim=1)))

    return intersection / (gt_pixels + pred_pixels - intersection)

def weighted_iou(preds, masks):
    #IoU, but the values are weighted with frequencies of pixels
    flat_preds = preds.argmax(dim=1).reshape(-1)
    flat_masks = masks.argmax(dim=1).reshape(-1)
    intersection = torch.sum(flat_preds == flat_masks)

    gt_pixels = torch.sum(torch.ones_like(masks.argmax(dim=1)))
    pred_pixels = torch.sum(torch.ones_like(preds.argmax(dim=1)))

    union = gt_pixels + pred_pixels - intersection

    class_freqs = torch.bincount(masks.argmax(dim=1).reshape(-1), minlength=3)  

    weights = 1 / (class_freqs.float() + 1e-10)  # Add 1e-10 to avoid division by zero
    weighted_iou = (intersection / union) * (weights[1] + weights[2])

    return weighted_iou


def eval_model(model, val_loader, wandb_experiment, criterion, device, epoch):

    validation_loss=0.0
    mean_acc_pixel=0.0
    mIoU=0.0
    fwIoU=0.0

    model.eval()
    n_val=len(val_loader)
    validation_loss=0
    id=0
    with tqdm(total=n_val, desc='Validation', unit='batch', leave=False) as pbar:
        columns=['id','image','guess','truth','epoch']
        pred_table=wandb.Table(columns=columns)

        for batch in val_loader:
            images=batch[0]
            masks=batch[1]


            images=images.to(device,dtype=torch.float32)
            masks=masks.to(device)

            with torch.no_grad():
                masks_predictions=model(images)
                loss=criterion(masks_predictions,masks)
                validation_loss+=loss.item()

                mean_acc_pixel+=mpa(masks,masks_predictions)
                mIoU+=iou(masks,masks_predictions)
                fwIoU+=weighted_iou(masks,masks_predictions)
                
                guess=masks_predictions[0]
                
                pred_table.add_data(id,wandb.Image(images[0]),wandb.Image(guess),wandb.Image(masks[0]),epoch)
                id+=1
                
                wandb_experiment.log(
                    {
                        'val loss': validation_loss/len(val_loader),
                        'mean acc pixel': mean_acc_pixel/len(val_loader),
                        'mIoU': mIoU/len(val_loader),
                        'fwIoU': fwIoU/len(val_loader),
                        'epoch': epoch
                    }
                )
            wandb_experiment.log({'predictions':pred_table})
            
            pbar.update()

    model.train()
    return mean_acc_pixel/len(val_loader)