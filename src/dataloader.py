import os
import torch
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import math
import torchvision
from torch.utils.data import DataLoader, random_split
from torch.nn.utils.rnn import pack_sequence, pad_sequence
import albumentations as A

seed = 24
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

class KvasirSEGDataset(Dataset):
    """
    Class provides with image and mask, or alternatively
    give an transformed/augmented version of these.
    """
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir) # list containing the names of the entries in the directory

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.images[index])
        image = np.array(Image.open(img_path, "r").convert("RGB"))
        mask = np.array(Image.open(mask_path, "r").convert("L"), dtype = np.float32) # greyskale = L in PIL
        mask[mask==255.0] = 1.0 # 255 decimal code for white, change this to 1 due to sigmoid on output.

        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]

        return image, mask


def data_loaders(batch_size, train_transforms, val_transforms, num_workers, pin_memory):
    """
    Get dataloaders of train, validation and test dataset. 
    Val transfomrs are also used on test data.

    """
    train_img_dir = "/home/feliciaj/data/Kvasir-SEG/train/train_images"
    train_mask_dir = "/home/feliciaj/data/Kvasir-SEG/train/train_images"
    val_img_dir = "/home/feliciaj/data/Kvasir-SEG/train/val_images"
    val_mask_dir = "/home/feliciaj/data/Kvasir-SEG/train/val_images"
    test_img_dir = "/home/feliciaj/data/Kvasir-SEG/train/test_images"
    test_mask_dir = "/home/feliciaj/data/Kvasir-SEG/train/test_images"

    train_ds = KvasirSEGDataset(
        image_dir = train_img_dir,
        mask_dir = train_mask_dir,
        transform = train_transforms,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size = batch_size,
        num_workers = num_workers,
        pin_memory = pin_memory,
        shuffle = True,
    )

    val_ds = KvasirSEGDataset(
        image_dir = val_img_dir,
        mask_dir = val_mask_dir,
        transform = val_transforms,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size = batch_size,
        num_workers = num_workers,
        pin_memory = False,
        shuffle = False,
    )
    
    test_ds = KvasirSEGDataset(
        image_dir = test_img_dir,
        mask_dir = test_mask_dor,
        transform = val_transforms
    )

    test_loader = DataLoader(
        test_ds,
        batch_size = batch_size,
        num_workers = num_workers,
        pin_memory = pin_memory,
        shuffle = False,
    ) 

    return train_loader, val_loader, test_loader



def move_images():
    """
    Function takes dataset containing image and corresponding mask, 
    and splits into folders for train, val and test data.

    Mask musth have equal filename as its corresponding image.
    """
    data_set = PolypDataset("/home/feliciaj/data/Kvasir-SEG/images/", "/home/feliciaj/data/Kvasir-SEG/masks/", transform=None)

    base_path = "/home/feliciaj/data/Kvasir-SEG/"

    dirs = [
        base_path+"train/train_images/", 
        base_path+"train/train_masks/", 
        base_path+"val/val_images/", 
        base_path+"val/val_masks/"
    ]
    for d in dirs:
        if not os.path.exists(d):
            os.makedirs(d)

    train_frac = 0.8
    val_frac = 0.2

    N = len(data_set) # number of images
    perm = np.random.permutation(N)
    filenames = np.array(os.listdir(base_path + "images/"))[perm] # shuffle filenames for randomness

    for i, f in enumerate(filenames):
        if (i < train_frac * N):
            middle_path = "train/train_"
        else:
            middle_path = "val/val_"

        # copy images
        shutil.copy(
            base_path + "images/" + f, 
            base_path + middle_path + "images/" + f
        )

        # copy masks
        shutil.copy(
            base_path + "masks/" + f, 
            base_path + middle_path + "masks/" + f
        )



if __name__ == "__main__":
    #move_images() # only run this once to split data
    PolypDataset(image_dir="/home/feliciaj/data/Kvasir-SEG/images/", mask_dir="/home/feliciaj/data/Kvasir-SEG/masks/")
   


    

