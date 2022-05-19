# test
import argparse

import wandb

import torch
import torchvision
import torchvision.models as models
from torch.utils.data import DataLoader
from torchvision import transforms

from .dataset import CustomDataset
from .model import EffUNet, EffUNet_B7, UNet
from .utils import load_checkpoint, wandb_image, area_ratio, iou_score
from .metric import BinaryMetrics

from crfseg import CRF

parser = argparse.ArgumentParser(description="baseline",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# 모델 관련 하이퍼 파라미터 parser
parser.add_argument("--image_size", type=int, dest="image_size")

# wandb 관련 parser
parser.add_argument("--wandb_project_name", type=str, dest="wandb_project_name", default="ex_project1") 
parser.add_argument("--wandb_entity_name", type=str, dest="wandb_entity_name", default="solov6") 
parser.add_argument("--wandb_name", type=str, dest="wandb_name", default="ex1") 

# 경로 관련 parser
parser.add_argument("--base_path", type=str, dest="base_path") 
parser.add_argument("--model_load_path", type=str, dest="model_load_path")

# SSL class 관련 parser
parser.add_argument("--class_name", type=str, dest="class_name") 

# model 관련 parser
parser.add_argument("--model", type=str, dest="model") 

args = parser.parse_args()

BASE_PATH = args.base_path
MODEL_LOAD_PATH = args.model_load_path

# Hyperparameters etc.
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_HEIGHT = args.image_size
IMAGE_WIDTH = args.image_size
PIN_MEMORY = True
NUM_WORKERS = 2

TEST_IMG_DIR = BASE_PATH + "/validation/images"
TEST_MASK_DIR = BASE_PATH + "/validation/masks"

WANDB_PROJECT = args.wandb_project_name 
WANDB_ENTITY = args.wandb_entity_name 
WANDB_NAME = args.wandb_name 

CLASS_NAME = args.class_name

MODEL = args.model

def main():
    run = wandb.init(project=WANDB_PROJECT, entity=WANDB_ENTITY, name=WANDB_NAME, reinit=True)

    test_transform = transforms.Compose([
                                        transforms.ToTensor(),
                                        transforms.Resize((IMAGE_HEIGHT,IMAGE_WIDTH)),
    ])

    mask_transform = transforms.Compose([
                                        transforms.ToTensor(),
                                        transforms.Resize((IMAGE_HEIGHT,IMAGE_WIDTH), torchvision.transforms.InterpolationMode.NEAREST),
    ])

    test_ds = CustomDataset(
        image_dir=TEST_IMG_DIR,
        mask_dir=TEST_MASK_DIR,
        image_transform=test_transform,
        mask_transform=mask_transform,
        class_name=CLASS_NAME
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=1,
        num_workers=2,
        pin_memory=False,
        shuffle=False,
    )
    
    # model
    if MODEL == 'eff-unet-b0':
        pre_model = models.efficientnet_b0(pretrained=True)
        pre_model_no_fc = list(pre_model.features.children())[:-1]
        model = EffUNet(3,1,pre_model_no_fc).to(DEVICE)
        print('model : Eff-Unet_B0')
    elif MODEL == 'eff-unet-b7':
        pre_model = models.efficientnet_b7(pretrained=True)
        pre_model_no_fc = list(pre_model.features.children())[:-1]
        model = EffUNet_B7(3,1,pre_model_no_fc).to(DEVICE)
        print('model : Eff-Unet_B7')
    elif MODEL == 'unet':
        model = UNet(1).to(DEVICE)
        print('model : Unet')
    elif MODEL == 'deeplabv3':
            model = models.segmentation.deeplabv3_resnet50(num_classes=1, pretrained=False).to(DEVICE)
            print('model : deeplabv3')

    # load model
    checkpoint = torch.load(MODEL_LOAD_PATH)
    load_checkpoint(checkpoint, model)
    model = model.to(DEVICE)

    with torch.no_grad():
        model.eval()

        cnt = 0
        total_iou = 0
        total_pixel_acc, total_dice, total_precision, total_recall = 0, 0, 0, 0

        for idx, (data, targets, _) in enumerate(test_loader):
            data = data.to(device=DEVICE)
            target = targets.to(device=DEVICE)

            # forward
            with torch.cuda.amp.autocast():
                pred = model(data)

            # crf
            crf = CRF(n_spatial_dims=2, requires_grad=False, smoothness_weight=1, smoothness_theta=1)
            pred = crf(pred)

            # iou score
            iou = iou_score(pred, target)
            total_iou += iou
            cnt += 1
            wandb.log({'iou': iou})

            metric = BinaryMetrics()
            pixel_acc, dice, precision, specificity, recall = metric(target, pred)
            total_pixel_acc += pixel_acc
            total_dice += dice
            total_precision += precision
            total_recall += recall
            wandb.log({
                "pixel_acc": pixel_acc,
                "dice": dice,
                "recall": recall,
                "precision": precision
            })

            target = target.cpu().numpy()
            pred = pred.cpu().numpy()

            # wandb 이미지 저장
            wandb_image(pred, target, data)

            # wandb area_ratio
            area_ratio(pred, CLASS_NAME)

        # wandb iou    
        mean_iou = total_iou / cnt

        # mean mIoU
        print(f'mean_iou: {mean_iou}')
        print(f"mean_dice: {total_dice / cnt}")
        print(f"mean_pixel_acc: {total_pixel_acc / cnt}")
        print(f"mean_recall: {total_recall / cnt}")
        print(f"mean_precision: {total_precision / cnt}")

        wandb.log({
            "mean_iou" : mean_iou,
            "mean_pixel_acc": total_pixel_acc / cnt,
            "mean_dice": total_dice / cnt,
            "mean_recall": total_recall / cnt,
            "mean_precision": total_precision / cnt
        })
        

if __name__ == "__main__":
    main()