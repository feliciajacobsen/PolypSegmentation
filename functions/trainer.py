import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from model import UNET


def train_model(loader, model, device, optimizer, criterion):
    """
    Function perform one epoch on entire dataset and outputs loss for each batch.

    Args:
        loader (object): iterable-style dataset.

        model (class): provides with a forward method.

        device (cuda object): cpu or gpu.

        optimizer (torch object): optimization algorithm. 

        criterion (torch oject): loss function with backward method.

    Returns:
        float: mean loss over batches
    """

    tqdm_loader = tqdm(loader) # make progress bar
    scaler = torch.cuda.amp.GradScaler() # gradient scaling to prevent undeflow

    model.train()
    losses = []

    for batch_idx, (data, targets) in enumerate(tqdm_loader):
        # move data and masks to same device as computed gradients
        data = data.to(device=device) 
        targets = targets.float().to(device=device)

        # mixed precision training
        with torch.cuda.amp.autocast():
            output = model(data) 
            loss = criterion(output, targets)

        # backprop
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # update tqdm loop
        tqdm_loader.set_postfix(loss=loss.item())
        
        # update loss
        losses.append(loss.item())





if __name__ == "__main__":
    pass
