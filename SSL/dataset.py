# dataset
import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np

class CustomDataset(Dataset):
    def __init__(self, image_dir, mask_dir, class_name, image_transform=None, mask_transform=None):
        self.image_dir = image_dir + f'/{class_name}/' # class image 주소 
        self.mask_dir = mask_dir + f'/{class_name}/' # class mask 주소
        
        self.image_transform = image_transform
        self.mask_transform = mask_transform

        self.images = os.listdir(self.image_dir) # class image 이름 list로 가져오기 

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        # image
        img_path = os.path.join(self.image_dir, self.images[index])
        image = np.array(Image.open(img_path).convert("RGB"), dtype=np.uint8)
        path_name = self.images[index]

        # mask
        mask_path = os.path.join(self.mask_dir, self.images[index])
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.uint8)
        mask = mask.reshape(mask.shape + (1,))

        if self.image_transform is not None:
            image = self.image_transform(image)
            mask = self.mask_transform(mask)

        return image, mask, path_name 

class EvalDataset(Dataset):
    def __init__(self, image_dir, class_name, image_transform=None):
        self.image_dir = image_dir + f'/{class_name}/' # class image 주소 
        
        self.image_transform = image_transform

        self.images = os.listdir(self.image_dir) # class image 이름 list로 가져오기 

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        # image
        img_path = os.path.join(self.image_dir, self.images[index])
        image = np.array(Image.open(img_path).convert("RGB"), dtype=np.uint8)
        path_name = self.images[index]

        if self.image_transform is not None:
            image = self.image_transform(image)

        return image, path_name