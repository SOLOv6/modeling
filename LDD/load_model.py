import torch.nn as nn
import torchvision.models as models
from LDD.model import EffUNet
from .model import UNet, EffUNet, EffUNet_B7

def load_model(model_name):
    if model_name == 'eff-unet-b0':
        pre_model = models.efficientnet_b0(pretrained=True)
        pre_model_no_fc = list(pre_model.features.children())[:-1]
        model = EffUNet(3, 3, pre_model_no_fc)

    elif model_name == 'eff-unet-b7':
        pre_model = models.efficientnet_b7(pretrained=True)
        pre_model_no_fc = list(pre_model.features.children())[:-1]
        model = EffUNet_B7(3, 3, pre_model_no_fc)

    elif model_name == 'unet':
        model = UNet(3)

    elif model_name == 'deeplabv3':
        model = models.segmentation.deeplabv3_mobilenet_v3_large(num_classes=21, pretrained=True)
        model.classifier[-1] = nn.Conv2d(256, 3, kernel_size=(1, 1), stride=(1, 1))
    
    print(f'model : {model_name}')

    return model