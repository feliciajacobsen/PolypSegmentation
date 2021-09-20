import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from model import UNET


def train_model(loader, model, device, optimizer, loss_function):
    """
    Function perform one epoch on entire dataset and outputs loss for each batch.

    Args:
        loader (object): iterable-style dataset.

        model (class): provides with a forward method.

        device (cuda object): cpu or gpu.

        optimizer (torch object): optimization algorithm. 

        loss_function (torch oject): loss function with backward method.

    Returns:
        float: mean loss over batches
    """

    tqdm_loader = tqdm(loader)

    model.train()
    losses = []

    for batch_idx, (data, targets) in enumerate(tqdm_loader):
        data = data.to(device=device) # move data to same device as gradients are computed
        targets = targets.float().to(device=device)

        with torch.cuda.amp.autocast():
            output = model(data)
            loss = loss_function(output, targets)

        optimizer.zero_grad()
        output = model(data)

        loss_function = criterion(data, targets)
        loss_function.backward()
        optimizer.step()
        losses.append(loss.item())




def main():
    pass




def run_model(root_dir, use_GPU=True):
    config = dict()
    config["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    config["lr"] = 1e-4
    config["batch_size"] = 64
    config["num_epochs"] = 3
    config["num_workers"] = 2
    config["num_classes"] = 1
    #config["image_height"] = 160 # 1280 originally
    #config["image_width"] = 240 # 1918 originally
    config["pin_memory"] = True
    config["load_model"] = False
    config["img_dir"] = "data/Kvasir-SEG/images/"
    config["mask_dir"] = "data/Kvasir-SEG/masks/"

if __name__ == "__main__":
    run_model()
