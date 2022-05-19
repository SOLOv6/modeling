import torch
import numpy as np
import wandb
import torch.nn as nn
from torchvision.ops.focal_loss import sigmoid_focal_loss

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

def check_accuracy(loader, model, pos_weight ,device="cuda"):
    """
    Check validation accuracy(iou)
    Do post-processing using threshold(default=.5)
    Logging WandB
    @param loader: Dataloader of current model
    @param loss_fn: BCEloss
    @param model: Unet or Eff-Unet-b*
    @param device: cuda or cpu
    """

    val_loss_em = 0
    cnt = 0

    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            preds = model(x)


            loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

            focal_loss = sigmoid_focal_loss(preds, y, reduction='mean')
            bce_loss = loss_fn(preds, y)

            loss = bce_loss * 0.8 + 5 * focal_loss

            val_loss_em += loss.item()
            cnt += 1

    # wandb
    wandb.log({'val_loss': val_loss_em / cnt})

    model.train()


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


def make_dict(pred, ratio, polygon):
    dic = {}
    if np.sum(pred) != 0:
        dic["is_damage"] = "true"
        dic["area_ratio"] = ratio
        dic["polygon"] = polygon

    else:
        dic["is_damage"] = "false"
        dic["area_ratio"] = ratio
        dic["polygon"] = polygon

    return dic



def mask2dic(pred, ratios, polygons):
    dent = pred.squeeze()[0]
    scratch = pred.squeeze()[1]
    spacing = pred.squeeze()[2]

    dent_dic = make_dict(dent, ratios[0], polygons[0])
    scratch_dic =make_dict(scratch, ratios[1], polygons[1])
    spacing_dic =make_dict(spacing, ratios[2], polygons[2])

    dic = {
        "dent": dent_dic,
        "scratch": scratch_dic,
        "spacing": spacing_dic
    }

    return dic

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

def miou(target, pred):
    intersection = np.logical_and(target, pred)
    union = np.logical_or(target, pred)
    miou_score = np.sum(intersection) / np.sum(union)
    return miou_score



