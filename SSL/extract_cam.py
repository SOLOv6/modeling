from torchvision.transforms import transforms
import torch
import numpy as np
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


def extract_cam_all(model, img, img_size, batch_size, confidence_threshold = 0.5, mode="toOne"):
    """
    extract CAM about all of category
    :param img: torch.tensor img
    :param img_size: resize img size
    :param batch_size: batch_size
    :param confidence_threshold: threshold of GT
    :param mode: to one
    :return: adding CAM img
    """

    transform = transforms.Compose([transforms.Resize((img_size, img_size)), ])
    img_tensor = transform(img)
    print("transform tensor shape", img_tensor.shape)

    # X, y
    target_layers = [model.resnet[-2]]  # resnet
    input_tensor = img_tensor

    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=True)

    targets_dent = [ClassifierOutputTarget(0)]  # 0: dent, 1: scratch, 2:spacing
    targets_scratch = [ClassifierOutputTarget(1)]  # 0: dent, 1: scratch, 2:spacing
    targets_spacing = [ClassifierOutputTarget(2)]  # 0: dent, 1: scratch, 2:spacing

    grayscale_cam_dent = cam(input_tensor=input_tensor, targets=targets_dent)
    grayscale_cam_scratch = cam(input_tensor=input_tensor, targets=targets_scratch)
    grayscale_cam_spacing = cam(input_tensor=input_tensor, targets=targets_spacing)

    conf = torch.sigmoid(model(input_tensor))


    res = torch.zeros((batch_size, 1, img_size, img_size))
    for batch in range(batch_size):

        dent_cam = grayscale_cam_dent[batch,:]
        scratch_cam = grayscale_cam_scratch[batch, :]
        spacing_cam = grayscale_cam_spacing[batch, :]

        # confidence 기준으로 버릴꺼 버리기기
        if conf[batch][0] < confidence_threshold: # dent
            dent_cam = np.where(dent_cam >= 0. , 0., 0.)
        elif conf[batch][1] < confidence_threshold: # scratch
            scratch_cam = np.where(dent_cam >= 0., 0., 0.)
        elif conf[batch][2] < confidence_threshold: # spacing
            spacing_cam = np.where(dent_cam >= 0., 0., 0.)


        if mode == "toOne":
            cam = torch.tensor(dent_cam + scratch_cam + spacing_cam).unsqueeze(0)

        res[batch] = cam

    img = torch.cat([img, res], dim= 1)

    print("FINISH!!!! CHECK SHAPE", img.shape)

    return img


def extract_cam_one(model, img, img_size, batch_size, confidence_threshold=0.5, class_name="dent"):
    transform = transforms.Compose([transforms.Resize((img_size, img_size)), ])
    img_tensor = transform(img)

    # X, y
    target_layers = [model.resnet[-2]]  # resnet
    input_tensor = img_tensor

    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=True)

    targets_dent = [ClassifierOutputTarget(0)]  # 0: dent, 1: scratch, 2:spacing
    targets_scratch = [ClassifierOutputTarget(1)]  # 0: dent, 1: scratch, 2:spacing
    targets_spacing = [ClassifierOutputTarget(2)]  # 0: dent, 1: scratch, 2:spacing

    grayscale_cam_dent = cam(input_tensor=input_tensor, targets=targets_dent)
    grayscale_cam_scratch = cam(input_tensor=input_tensor, targets=targets_scratch)
    grayscale_cam_spacing = cam(input_tensor=input_tensor, targets=targets_spacing)

    ###################################################################
    input_tensor = input_tensor.to("cuda")
    conf = torch.sigmoid(model(input_tensor).cpu().detach())
    ###############################################################

    if class_name == "dent":
        res = concat_cam(grayscale_cam_dent, img_size, batch_size, conf, confidence_threshold)

    elif class_name == "scratch":
        res = concat_cam(grayscale_cam_scratch, img_size, batch_size, conf, confidence_threshold)

    else:
        res = concat_cam(grayscale_cam_spacing, img_size, batch_size, conf, confidence_threshold)


    return res 

def concat_cam(cams, img_size, batch_size, conf, confidence_threshold):
    res = torch.zeros((batch_size, 1, img_size, img_size))
    for batch in range(batch_size):
        cam = cams[batch, :]

        # confidence 기준으로 버릴꺼 버리기기
        if conf[batch][0] < confidence_threshold:
            cam = np.where(cam >= 0., 0., 0.)

        tensor_cam = torch.tensor(cam).unsqueeze(0)

        res[batch] = tensor_cam

    return res









