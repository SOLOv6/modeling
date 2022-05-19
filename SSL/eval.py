# test
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

import wandb

from .dataset import EvalDataset
from .load_model import load_model
from .utils import load_checkpoint, area_ratio, save_result

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
parser.add_argument("--save_result_dir", type=str, dest="save_result_dir") 

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

EVAL_IMG_DIR = BASE_PATH + "/train/images"
SAVE_RESULT_DIR = args.save_result_dir

WANDB_PROJECT = args.wandb_project_name 
WANDB_ENTITY = args.wandb_entity_name 
WANDB_NAME = args.wandb_name 

CLASS_NAME = args.class_name

MODEL = args.model


def main():
    wandb.init(project=WANDB_PROJECT, entity=WANDB_ENTITY, name=WANDB_NAME, reinit=True)

    test_transform = transforms.Compose([
                                        transforms.ToTensor(),
                                        transforms.Resize((IMAGE_HEIGHT,IMAGE_WIDTH)),
    ])
    eval_ds = EvalDataset(
        image_dir=EVAL_IMG_DIR,
        image_transform=test_transform,
        class_name=CLASS_NAME
    )
    eval_loader = DataLoader(
        eval_ds,
        batch_size=1,
        num_workers=2,
        pin_memory=False,
        shuffle=False,
    )

    # load model
    model = load_model(MODEL)
    checkpoint = torch.load(MODEL_LOAD_PATH)
    load_checkpoint(checkpoint, model)
    model = model.to(DEVICE)

    # eval
    with torch.no_grad():
        model.eval()

        for idx, (data, path_name) in enumerate(eval_loader):
            data = data.to(device=DEVICE)

            # forward
            with torch.cuda.amp.autocast():
                pred = torch.sigmoid(model(data))

                pred[pred >= 0.5] = 1.0
                pred[pred < 0.5] = 0.0 

            # sample for mask iou
            pred = pred.cpu().numpy()

            # wandb area_ratio
            area_ratio(pred, CLASS_NAME)

            # save_result_image
            save_path = SAVE_RESULT_DIR 
            save_result(save_path, WANDB_NAME, path_name, pred)

if __name__ == "__main__":
    main()