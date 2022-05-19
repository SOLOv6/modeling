# dataloader
from torch.utils.data import DataLoader
from dataset import CustomDataset, ValidationDataset


def dataloaders(
    train_dir,
    train_maskdir,
    val_dir,
    val_maskdir,
    batch_size,
    train_transform,
    val_transform,
    mask_transform,
    num_workers=4,
    pin_memory=True,
):
    train_ds = CustomDataset(
        image_dir=train_dir,
        mask_dir=train_maskdir,
        image_transform=train_transform,
        mask_transform=mask_transform,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
        drop_last=True
    )

    val_ds = ValidationDataset(
        image_dir=val_dir,
        mask_dir=val_maskdir,
        image_transform=train_transform,
        mask_transform=mask_transform,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
        drop_last=True
    )

    return train_loader, val_loader