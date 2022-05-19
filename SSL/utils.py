import os
import numpy as np
import wandb
import cv2

import torch
import torch.nn as nn

from torchvision.ops.focal_loss import sigmoid_focal_loss
from crfseg import CRF

def save_checkpoint(state, wandb_id, filename="tr_model_v1_checkpoint.pth"):
    """
    saved trained model in filename(path)
    @param state: saved model
    @param wandb_id: wandb run for logging
    @param filename: path for saved model
    """
    print("=> Saving checkpoint")
    torch.save(state, filename)

    artifact = wandb.Artifact('model', type='model')
    artifact.add_file(filename)

    wandb_id.log_artifact(artifact)
    wandb_id.join()


def load_checkpoint(checkpoint, model):
    """
    Load saved model in checkpoint
    @param checkpoint: path of saved model
    @param model: Deep learning network
    """
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])


def check_accuracy(loader, model, loss_fn, device="cuda"):
    """
    Check validation accuracy(iou)
    Do post-processing using threshold(default=.5)
    Logging WandB
    @param loader: Dataloader of current model
    @param loss_fn: BCEloss + Focalloss
    @param model: Unet or Eff-Unet-b*
    @param device: cuda or cpu
    """

    val_loss_em = 0
    cnt = 0

    model.eval()
    with torch.no_grad():
        for x, y, _ in loader:
            x = x.to(device)
            y = y.to(device)

            preds = model(x)
            crf = CRF(n_spatial_dims=2, requires_grad=False, smoothness_weight=0.75, smoothness_theta=0.75)
            predictions = crf(preds)

            # both loss
            loss = loss_fn.both_loss(predictions, y)
            
            val_loss_em += loss.item()
            cnt += 1

    # wandb
    wandb.log({'val_loss': val_loss_em / cnt})

    model.train()
    return val_loss_em / cnt


def wandb_image(pred, target, data):
    """
    Logging WandB Image for Compare pred
    @param pred: pred from model
    @param target: GT
    @param data: original data
    """
    label = {0: 'False', 1: 'True'}

    pred_wandb = pred.squeeze()
    target_wandb = target.squeeze()

    pred_wandb[pred_wandb >= 0.5] = 1
    pred_wandb[pred_wandb < 0.5] = 0

    target_wandb[target_wandb >= 0.5] = 1
    target_wandb[target_wandb < 0.5] = 0

    images = wandb.Image(data, masks={
        'predictions': {
            'mask_data': pred_wandb,
            'class_labels': label
        },
        'ground truth': {
            'mask_data': target_wandb,
            'class label': label
        }
    })
    wandb.log({'Eff-UNet': images})


def area_ratio(pred, class_name):
    """
    Calculate area of segment
    @param pred: pred from model
    @param class_name: "Dent" or "Scratch" or "Spacing
    @return:
    """
    ratio = np.sum(pred.squeeze()) / (224 * 224)

    wandb.log({f'{class_name}_ratio': ratio})
    return ratio

def save_result(save_path, test_model_name, image_name, pred):
    """
    Saving pred in save_path
    @param save_path: save_path
    @param test_model_name: folder name
    @param image_name: original image name
    @param pred: pred from model
    """
    save_folder = os.path.join(save_path, test_model_name)

    if not os.path.exists(save_folder):
        os.makedirs(os.path.join(save_folder))

    img_name = os.path.join(save_folder, *image_name)
    pred = np.repeat(pred.squeeze(0).astype(np.int32), repeats=3, axis=0)

    pred = pred.transpose(1, 2, 0)
    pred = np.where(pred >= 1, 255, 0)

    cv2.imwrite(img_name, pred)

def iou_score(output, target):
    """
    Calculate IoU
    @param target: GT
    @param pred: pred from model
    @return: mask iou score
    """
    smooth = 1e-5

    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
    output_ = output > 0.5
    target_ = target > 0.5
    intersection = (output_ & target_).sum()
    union = (output_ | target_).sum()

    return (intersection + smooth) / (union + smooth)


def iou_pytorch(outputs: torch.Tensor, labels: torch.Tensor):
    """
    Calculate IoU
    @param target: GT
    @param pred: pred from model
    @return: mask iou score
    """
    SMOOTH = 1e-6
    # You can comment out this line if you are passing tensors of equal shape
    # But if you are passing output from UNet or something it will most probably
    # be with the BATCH x 1 x H x W shape
    outputs = outputs.squeeze(1)  # BATCH x 1 x H x W => BATCH x H x W
    
    intersection = (outputs & labels).float().sum((1, 2))  # Will be zero if Truth=0 or Prediction=0
    union = (outputs | labels).float().sum((1, 2))         # Will be zzero if both are 0
    
    iou = (intersection + SMOOTH) / (union + SMOOTH)  # We smooth our devision to avoid 0/0
    
    thresholded = torch.clamp(20 * (iou - 0.5), 0, 10).ceil() / 10  # This is equal to comparing with thresolds
    
    return thresholded  # Or thresholded.mean() if you are interested in average across the batch

# if __name__ == "__main__":


#     img = np.random.randint(0, 255, (1,224,224))
#     print(img.shape)
#     img = np.repeat(img, repeats=3, axis=0)
#     img.astype(np.float32)
#     print(img.shape)
#     print(img[0][0][:10])
#     imgs = img[0][0][:10]
#     print(np.where(imgs > 100, 255, 0))

#     # cv2.imwrite("./save_SSL/ex.jpg", img)