# train
import argparse

import torch 
import torch.nn as nn
import torchvision
from torchvision import transforms
from tqdm import tqdm
import torch.optim as optim

import wandb

from .loss import Loss_Functions
from .load_model import load_model
from .gdc_model import ResnetGradCam
from .dataloader import dataloaders
from .utils import check_accuracy, save_checkpoint
from .extract_cam import extract_cam_one

from crfseg import CRF


parser = argparse.ArgumentParser(description="baseline",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# 모델 관련 하이퍼 파라미터 parser
parser.add_argument("--lr", default=1e-3, type=float, dest="lr")
parser.add_argument("--batch_size", default=4, type=int, dest="batch_size")
parser.add_argument("--num_epoch", default=100, type=int, dest="num_epoch")
parser.add_argument("--optimizer", default="Adam", type=str, dest="optimizer") 
parser.add_argument("--image_size", type=int, dest="image_size")
parser.add_argument("--weight_decay", type=float, dest="weight_decay", default=0.)

# wandb 관련 parser
parser.add_argument("--wandb_project_name", type=str, dest="wandb_project_name", default="ex_project1") 
parser.add_argument("--wandb_entity_name", type=str, dest="wandb_entity_name", default="solov6") 
parser.add_argument("--wandb_name", type=str, dest="wandb_name", default="ex1") 

# 경로 관련 parser
parser.add_argument("--base_path", type=str, dest="base_path") 
parser.add_argument("--model_save_name", type=str, dest="model_save_name") 

# SSL class 관련 parser
parser.add_argument("--class_name", type=str, dest="class_name") 

# model 관련 parser
parser.add_argument("--model", type=str, dest="model") 
parser.add_argument("--pretrained_path", type=str, dest="pretrained_path",  default=None) 
parser.add_argument("--gdc_model_path", type=str, dest="gdc_model_path",  default=None) 

# loss function parser
parser.add_argument("--loss_fn", type=str, dest="loss_fn") 
parser.add_argument("--focal_alpha", type=float, dest="focal_alpha", default=0.0) 
parser.add_argument("--focal_gamma", type=float, dest="focal_gamma", default=0.0) 
parser.add_argument("--both_weight_focal", type=float, dest="both_weight_focal", default=0.0) 
parser.add_argument("--both_weight_gamma", type=float, dest="both_weight_gamma", default=0.0) 

args = parser.parse_args()

BASE_PATH = args.base_path
MODEL_SAVE_NAME = args.model_save_name 

# Hyperparameters etc.
LEARNING_RATE = args.lr
DEVICE = "cuda" if torch.cuda.is_available() else "cpu" 
BATCH_SIZE = args.batch_size
OPTIMIZER = args.optimizer
NUM_EPOCHS = args.num_epoch
IMAGE_HEIGHT = args.image_size
IMAGE_WIDTH = args.image_size
WEIGHT_DECAY = args.weight_decay 
PIN_MEMORY = True
NUM_WORKERS = 2

TRAIN_IMG_DIR = BASE_PATH + "/train/images"
TRAIN_MASK_DIR = BASE_PATH + "/train/masks"
VAL_IMG_DIR = BASE_PATH + "/validation/images"
VAL_MASK_DIR = BASE_PATH + "/validation/masks"

WANDB_PROJECT = args.wandb_project_name 
WANDB_ENTITY = args.wandb_entity_name 
WANDB_NAME = args.wandb_name 

CLASS_NAME = args.class_name

MODEL = args.model
PRETRAIN_PATH = args.pretrained_path
GDC_MODEL_PATH = args.gdc_model_path

LOSS_FN = args.loss_fn
FOCAL_ALPHA = args.focal_alpha
FOCAL_GAMMA = args.focal_gamma
BOTH_WEIGHT_FOCAL = args.both_weight_focal
BOTH_WEIGHT_GAMMA = args.both_weight_gamma


config = {
    "EPOCHS": NUM_EPOCHS,
    "BATCH_SIZE": BATCH_SIZE,
    "LEARNING_RATE": LEARNING_RATE,
    "IMAGE_HEIGHT": IMAGE_HEIGHT,
    "IMAGE_WIDTH": IMAGE_WIDTH,
    "OPTIMIZER": OPTIMIZER,
    "WEIGHT_DECAY": WEIGHT_DECAY 
}


def main():
    # wandb
    run = wandb.init(project=WANDB_PROJECT, entity=WANDB_ENTITY, name=WANDB_NAME, config=config, reinit=True) 

    train_transform = transforms.Compose([
                                        transforms.ToTensor(),
                                        transforms.Resize((IMAGE_HEIGHT,IMAGE_WIDTH)),
    ])

    mask_transform = transforms.Compose([
                                        transforms.ToTensor(),
                                        transforms.Resize((IMAGE_HEIGHT,IMAGE_WIDTH), torchvision.transforms.InterpolationMode.NEAREST)
    ])

    val_transform = transforms.Compose([
                                      transforms.ToTensor(),
                                      transforms.Resize((IMAGE_HEIGHT,IMAGE_WIDTH)),
    ])
    
    # model
    model = load_model(MODEL).to(device=DEVICE)

    # loss
    loss_fn = Loss_Functions(focal_alpha = FOCAL_ALPHA,
                             focal_gamma = FOCAL_GAMMA,
                             both_weight_focal = BOTH_WEIGHT_FOCAL,
                             both_weight_bce =  BOTH_WEIGHT_GAMMA)
    
    # optimizer
    if OPTIMIZER == "Adam": 
        optimizer = optim.Adam(model.parameters(),
                             lr=LEARNING_RATE,
                             weight_decay=WEIGHT_DECAY)
    elif OPTIMIZER == "AdamW":
        optimizer = optim.AdamW(model.parameters(),
                                lr=LEARNING_RATE,
                                weight_decay=WEIGHT_DECAY)

    elif OPTIMIZER == "StepLR":
        optimizer = optim.SGD(model.parameters(),
                              lr=LEARNING_RATE,
                              weight_decay=WEIGHT_DECAY) 
        scheduler = optim.lr_scheduler.StepLR(optimizer,
                                              step_size=10,
                                              gamma=0.75)

    train_loader, val_loader = dataloaders(
        TRAIN_IMG_DIR,
        TRAIN_MASK_DIR,
        VAL_IMG_DIR,
        VAL_MASK_DIR,
        BATCH_SIZE,
        train_transform,
        val_transform,
        mask_transform,
        CLASS_NAME,
        NUM_WORKERS,
        PIN_MEMORY,
    )

    # load Global Damage Detector model
    gdc_model = ResnetGradCam()
    loaded = torch.load(GDC_MODEL_PATH)
    gdc_model.load_state_dict(loaded, strict=False)

    # CRF(Conditional Random Field)
    crf = CRF(n_spatial_dims=2, requires_grad=False, smoothness_weight=0.75, smoothness_theta=0.75)

    # train
    scaler = torch.cuda.amp.GradScaler()
    for epoch in range(NUM_EPOCHS):
        # wandb
        wandb.watch(model, log="all", log_freq=epoch)

        print(f'\n----------- epoch : {epoch+1} --------------------')
        loop = tqdm(train_loader)

        loss_epoch = 0
        cnt = 0
        for batch_idx, (data, targets, _) in enumerate(loop):
            cam_img = extract_cam_one(gdc_model, data, IMAGE_HEIGHT, BATCH_SIZE, confidence_threshold=0.1, class_name=CLASS_NAME) ### parser 

            cam_img = cam_img.to(device=DEVICE)
            data = data.to(device=DEVICE)
            targets = targets.float().to(device=DEVICE)

            # forward
            with torch.cuda.amp.autocast():
                predictions = crf(model(data))

                if LOSS_FN == 'bce_loss':
                    loss = loss_fn.bce_loss(predictions, targets, BATCH_SIZE, IMAGE_HEIGHT)
                elif LOSS_FN == 'focal_loss':
                    loss = loss_fn.focal_loss(predictions, targets)
                elif LOSS_FN == "both_loss":
                    loss = loss_fn.both_loss(predictions, targets)
                elif LOSS_FN == 'cam_loss':
                    loss = loss_fn.cam_loss(predictions, targets, cam_img)

            loss_epoch += loss.item()
            cnt += 1

            # backward
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # update tqdm loop
            loop.set_postfix(loss=loss.item())

            # wandb
            wandb.log({'train_batch_loss': loss.item()})

        # wandb
        wandb.log({'train_loss': loss_epoch/cnt})

        # check accuracy
        check_accuracy(val_loader, model, loss_fn, device=DEVICE) 

    # save model
    model = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }

    save_checkpoint(model, run, MODEL_SAVE_NAME)

if __name__ == "__main__":
    main()

