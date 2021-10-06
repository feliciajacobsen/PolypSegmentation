import os
import torch
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import math
from torch.utils.data import DataLoader, random_split
from torch.nn.utils.rnn import pack_sequence, pad_sequence


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



def data_loader(frac, batch_size, num_workers, pin_memory, transform):
    data_set = PolypDataset("/home/feliciaj/data/Kvasir-SEG/images/", "/home/feliciaj/data/Kvasir-SEG/masks/", transform)

    train_size = math.floor(len(data_set)*frac)
    test_size = len(data_set) - train_size  
    
    train_set, test_set = random_split(data_set, [train_size, test_size])

    def custom_collate_pad(batch):
        data = pad_sequence([item[0] for item in batch])
        target = pad_sequence([item[1] for item in batch])
        return torch.Tensor(data), torch.Tensor(target)


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


if __name__ == "__main__":
    PolypDataset(image_dir="/home/feliciaj/data/Kvasir-SEG/images/", mask_dir="/home/feliciaj/data/Kvasir-SEG/masks/")
    #train_loader, val_loader = data_loader(0.8, 64, num_workers=1, pin_memory=False)


    

