import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim

from Unet import UNET
from double_Unet import double_UNET, double_UNET_2
from resunetplusplus import Res_Unet_Plus_Plus

from dataloader import data_loader
from utils import check_accuracy, load_checkpoint, save_checkpoint, save_preds_as_imgs


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
        None
    """           

    tqdm_loader = tqdm(loader) # make progress bar
    scaler = torch.cuda.amp.GradScaler() # gradient scaling to prevent undeflow

    model.train()
    losses = []

    for batch_idx, (data, targets) in enumerate(tqdm_loader):
        # move data and masks to same device as computed gradients
        data = data.to(device=device) 
        targets = targets.float().unsqueeze(1).to(device=device) # add on channel dimension of 1

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

def run_model():
    config = dict()
    config["lr"] = 1e-4
    config["device"] = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    config["load_model"] = False
    config["num_epochs"] = 300
    config["numcl"] = 1
    config["batch_size"] = 64
    config["pin_memory"] = True
    config["num_workers"] = 1
    config["image_height"] = 240
    config["image_width"] = 240
   
    train_transforms = A.Compose(
        [   A.Resize(height=config["image_height"], width=config["image_width"]),
            A.Rotate(limit=35, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.Normalize(
                mean=[0.5579, 0.3214, 0.2350],
                std=[0.3185, 0.2218, 0.1875],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

    val_transforms = A.Compose(
        [   
            A.Resize(height=config["image_height"], width=config["image_width"]),
            A.Normalize(
                mean=[0.5579, 0.3214, 0.2350],
                std=[0.3185, 0.2218, 0.1875],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )
    
   
    #model = UNET(in_channels=3, out_channels=config["numcl"]).to(config["device"])
    model = Res_Unet_Plus_Plus(in_channels=3).to(config["device"])
    #model = double_UNET_2(in_channels=3, out_channels=64).to(config["device"])
    criterion = nn.BCEWithLogitsLoss() #  Sigmoid layer and the BCELoss
    optimizer = optim.Adam(model.parameters(), lr=config["lr"])

    train_loader, val_loader = data_loader(
        batch_size=config["batch_size"], 
        num_workers=config["num_workers"], 
        pin_memory=config["pin_memory"], 
        transform=val_transforms
    )
    
    
    if config["load_model"]:
        load_checkpoint(torch.load("my_checkpoint.pt"), model)
    
    check_accuracy(val_loader, model, device=config["device"])


    for epoch in range(config["num_epochs"]):
        train_model(train_loader, model, config["device"], optimizer, criterion)

        # save model
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer" : optimizer.state_dict()
        }
        save_checkpoint(checkpoint)

        # check accuracy
        check_accuracy(val_loader, model, device=config["device"])

        # print examples to a folder
        save_preds_as_imgs(
            val_loader, model, folder="/home/feliciaj/data/Kvasir-SEG/resunetplusplus_saved_images/", device=config["device"]
        )



if __name__ == "__main__":
    run_model()