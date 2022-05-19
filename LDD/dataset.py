# dataset
import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np


class CustomDataset(Dataset):
    def __init__(self, image_dir, mask_dir, image_transform=None, mask_transform=None):
        self.dent_scratch_image_dir = image_dir + '/dent/' # dent, scratch image 폴더 주소
        self.spacing_image_dir = image_dir + '/spacing/' # spacing image 폴더 주소

        self.dent_mask_dir = mask_dir + '/dent/' # dent mask 주소 
        self.scratch_mask_dir = mask_dir + '/scratch/' # scratch mask 주소
        self.spacing_mask_dir = mask_dir + '/spacing/' # spacing mask 주소

        self.pseudo_dent_mask_dir = '/content/drive/Shareddrives/SOCAR_PROJECT_DATA/SSL/save_result_final' + '/make_dent_mask/'
        self.pseudo_scratch_mask_dir = '/content/drive/Shareddrives/SOCAR_PROJECT_DATA/SSL/save_result_final' + '/make_scratch_mask/'
        self.pseudo_spacing_mask_dir = '/content/drive/Shareddrives/SOCAR_PROJECT_DATA/SSL/save_result_final' + '/make_spacing_mask/'

        self.image_transform = image_transform
        self.mask_transform = mask_transform

        self.dent_scratch_images = os.listdir(self.dent_scratch_image_dir) # dent,scratch image이름 list로 가져오기
        self.spacing_images = os.listdir(self.spacing_image_dir) # spacing image 이름 list로 가져오기 
        self.images = self.dent_scratch_images + self.spacing_images # 전체 이미지 이름 list (총 3342장) 

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        if index < len(self.dent_scratch_images):
            # image
            img_path = os.path.join(self.dent_scratch_image_dir, self.images[index])
            image = np.array(Image.open(img_path).convert("RGB"), dtype=np.uint8)

            # dent mask
            dent_mask_path = os.path.join(self.dent_mask_dir, self.images[index]) 
            dent_mask = np.array(Image.open(dent_mask_path).convert("L"), dtype=np.uint8)
            dent_mask = dent_mask.reshape(dent_mask.shape + (1,))

            # scratch mask
            scratch_mask_path = os.path.join(self.scratch_mask_dir, self.images[index])
            scratch_mask = np.array(Image.open(scratch_mask_path).convert("L"), dtype=np.uint8)
            scratch_mask = scratch_mask.reshape(scratch_mask.shape + (1,))

            # spacing mask
            spacing_mask_path = os.path.join(self.pseudo_spacing_mask_dir, self.images[index])
            spacing_mask = np.array(Image.open(scratch_mask_path).convert("L"), dtype=np.uint8)
            spacing_mask.resize((dent_mask.shape[0],dent_mask.shape[1]))
            spacing_mask = spacing_mask.reshape(spacing_mask.shape + (1,))

            # total mask
            mask = np.concatenate([dent_mask, scratch_mask, spacing_mask], axis=2)
            
        else:
            # image
            img_path = os.path.join(self.spacing_image_dir, self.images[index])
            image = np.array(Image.open(img_path).convert("RGB"), dtype=np.uint8)

            # spacing mask
            spacing_mask_path = os.path.join(self.spacing_mask_dir, self.images[index])
            spacing_mask = np.array(Image.open(spacing_mask_path).convert("L"), dtype=np.uint8)
            spacing_mask = spacing_mask.reshape(spacing_mask.shape + (1,))

            # dent mask
            dent_mask_path = os.path.join(self.pseudo_dent_mask_dir, self.images[index]) 
            dent_mask = np.array(Image.open(dent_mask_path).convert("L"), dtype=np.uint8)
            dent_mask.resize((spacing_mask.shape[0],spacing_mask.shape[1]))
            dent_mask = dent_mask.reshape(dent_mask.shape + (1,))

            # scratch mask
            scratch_mask_path = os.path.join(self.pseudo_scratch_mask_dir, self.images[index])
            scratch_mask = np.array(Image.open(scratch_mask_path).convert("L"), dtype=np.uint8)
            scratch_mask.resize((spacing_mask.shape[0],spacing_mask.shape[1]))
            scratch_mask = scratch_mask.reshape(scratch_mask.shape + (1,))

            # total mask
            mask = np.concatenate([dent_mask, scratch_mask, spacing_mask], axis=2)

        if self.image_transform is not None:
            image = self.image_transform(image)
            mask = self.mask_transform(mask)

        return image, mask


class ValidationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, image_transform=None, mask_transform=None):
        self.dent_scratch_image_dir = image_dir + '/dent/' # dent, scratch image 폴더 주소
        self.spacing_image_dir = image_dir + '/spacing/' # spacing image 폴더 주소

        self.dent_mask_dir = mask_dir + '/dent/' # dent mask 주소 
        self.scratch_mask_dir = mask_dir + '/scratch/' # scratch mask 주소
        self.spacing_mask_dir = mask_dir + '/spacing/' # spacing mask 주소

        self.image_transform = image_transform
        self.mask_transform = mask_transform

        self.dent_scratch_images = os.listdir(self.dent_scratch_image_dir) # dent,scratch image이름 list로 가져오기
        self.spacing_images = os.listdir(self.spacing_image_dir) # spacing image 이름 list로 가져오기 
        self.images = self.dent_scratch_images + self.spacing_images # 전체 이미지 이름 list (총 3342장) 

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        if index < len(self.dent_scratch_images):
            # image
            img_path = os.path.join(self.dent_scratch_image_dir, self.images[index])
            image = np.array(Image.open(img_path).convert("RGB"), dtype=np.uint8)

            # dent mask
            dent_mask_path = os.path.join(self.dent_mask_dir, self.images[index]) 
            dent_mask = np.array(Image.open(dent_mask_path).convert("L"), dtype=np.uint8)
            dent_mask = dent_mask.reshape(dent_mask.shape + (1,))

            # scratch mask
            scratch_mask_path = os.path.join(self.scratch_mask_dir, self.images[index])
            scratch_mask = np.array(Image.open(scratch_mask_path).convert("L"), dtype=np.uint8)
            scratch_mask = scratch_mask.reshape(scratch_mask.shape + (1,))

            # spacing mask
            spacing_mask = np.zeros_like(dent_mask, dtype=np.uint8)

            # total mask
            mask = np.concatenate([dent_mask, scratch_mask, spacing_mask], axis=2)
            
        else:
            # image
            img_path = os.path.join(self.spacing_image_dir, self.images[index])
            image = np.array(Image.open(img_path).convert("RGB"), dtype=np.uint8)

            # spacing mask
            spacing_mask_path = os.path.join(self.spacing_mask_dir, self.images[index])
            spacing_mask = np.array(Image.open(spacing_mask_path).convert("L"), dtype=np.uint8)
            spacing_mask = spacing_mask.reshape(spacing_mask.shape + (1,))

            # dent mask
            dent_mask = np.zeros_like(spacing_mask, dtype=np.uint8)

            # scratch mask
            scratch_mask = np.zeros_like(spacing_mask, dtype=np.uint8)
            
            # total mask
            mask = np.concatenate([dent_mask, scratch_mask, spacing_mask], axis=2)

        if self.image_transform:
            image = self.image_transform(image)
            mask = self.mask_transform(mask)

        return image, mask