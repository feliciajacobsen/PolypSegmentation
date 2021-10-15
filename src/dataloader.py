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

class PolypDataset(Dataset):
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



def data_loader(batch_size, num_workers, pin_memory, transform):
    data_set = PolypDataset("/home/feliciaj/data/Kvasir-SEG/images/", "/home/feliciaj/data/Kvasir-SEG/masks/", transform)
    frac = 0.8
    train_size = math.floor(len(data_set)*frac)
    test_size = len(data_set) - train_size  
    
    train_set, test_set = random_split(data_set, [train_size, test_size], generator=torch.Generator().manual_seed(seed))

    train_loader = DataLoader(
        train_set,
        batch_size = batch_size,
        num_workers = num_workers,
        pin_memory = pin_memory,
        shuffle = True,
    )
    
    val_loader = DataLoader(
        test_set,
        batch_size = batch_size,
        num_workers = num_workers,
        pin_memory = pin_memory,
        shuffle = True,
    )
    
    return train_loader, val_loader


def store_split_data(loader, folder):
    """
    Function saves the predicted masks an stores in separate folder.
    """
    """
    path = os.path.join(folder, "val_images")
    os.makedirs(path)
    path = os.path.join(folder, "val_masks")
    os.makedirs(path)
    """
    # must get filename in some way

    for idx, (img, mask) in enumerate(loader):
        torchvision.utils.save_image(img, f"{folder}/val_images/{filename}.png")
        torchvision.utils.save_image(mask, f"{folder}/val_masks/{filename}.png")



if __name__ == "__main__":
    PolypDataset(image_dir="/home/feliciaj/data/Kvasir-SEG/images/", mask_dir="/home/feliciaj/data/Kvasir-SEG/masks/")
    #train_loader, val_loader = data_loader(64, num_workers=1, pin_memory=False, transform=transform)
    #store_split_data(val_loader, "/home/feliciaj/data/Kvasir-SEG/")
    #store_split_data(train_loader, "/home/feliciaj/data/Kvasir-SEG/")


    

