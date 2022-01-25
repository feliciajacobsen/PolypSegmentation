import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Local imports
from unet import UNet, UNet_dropout
from resunetplusplus import Res_Unet_Plus_Plus
from doubleunet import DoubleUNet
from dataloader import data_loaders
from utils import (
    check_scores, 
    load_checkpoint, 
    save_checkpoint, 
    save_preds_as_imgs,  
    EarlyStopping
)

from metrics import (
    BCEDiceLoss,
    DiceLoss
)


def train_model(loader, model, device, optimizer, criterion):
    """
    Function perform one epoch on entire dataset and prints loss for each batch.

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
    scaler = torch.cuda.amp.GradScaler()

    model.train()
    
    losses = []
    for batch_idx, (data, targets) in enumerate(tqdm_loader):
        # move data and masks to same device as computed gradients
        data = data.to(device=device) 
        targets = targets.float().unsqueeze(1).to(device=device) # add on channel dimension of 1

        with torch.cuda.amp.autocast():
            output = model(data) 
            loss = criterion(output, targets)

        # update loss
        losses.append(loss.item())

        # backprop
        optimizer.zero_grad() # zero out previous gradients
        scaler.scale(loss).backward() # scale loss before backprop
        scaler.step(optimizer) # update gradients
        scaler.update() # update scale factor
        
        """
        if scheduler is not None:
            scheduler.step(mean_loss)
        """
        # update tqdm loop
        tqdm_loader.set_postfix(loss=loss.item())

    return sum(losses)/len(loader)
        

def run_model():
    config = dict()
    config["lr"] = 1e-4
    config["device"] = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    config["load_model"] = False
    config["num_epochs"] = 3
    config["numcl"] = 1
    config["batch_size"] = 64
    config["pin_memory"] = True
    config["num_workers"] = 4
    config["image_height"] = 256
    config["image_width"] = 256
    config["model_name"] = "unet"
    config["save_folder"] = "/home/feliciaj/PolypSegmentation/saved_models/" + config["model_name"] + "/"
   
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
    #model = Res_Unet_Plus_Plus(in_channels=3).to(config["device"])
    model = DoubleUNet().to(config["device"])

    #criterion = nn.BCEWithLogitsLoss() #  Sigmoid layer and the BCELoss
    #criterion = BCEDiceLoss() # Sigmoid layer and Dice loss
    criterion = DiceLoss()

    optimizer = optim.Adam(model.parameters(), lr=config["lr"])

    #scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=20, verbose=True) 

    early_stopping = None #EarlyStopping()

    train_loader, val_loader, test_loader = data_loaders(
        batch_size=config["batch_size"], 
        num_workers=config["num_workers"], 
        pin_memory=config["pin_memory"], 
        transform=val_transforms
    )

    
    if config["load_model"]:
        load_checkpoint(torch.load("./checkpoint_1.pt"), model)
    #check_scores(val_loader, model, device=config["device"])

    val_epoch_loss = []
    train_epoch_loss = []

    for epoch in range(config["num_epochs"]):
        # train on training data, prints accuracy and dice score of training data
        mean_train_loss = train_model(train_loader, model, config["device"], optimizer, criterion)

        # save model
        if epoch == config["num_epochs"]-1:
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer" : optimizer.state_dict()
            }
            save_checkpoint(epoch, checkpoint, config["save_folder"]+"checkpoint_1.pt")

        # check validation loss
        mean_val_loss = check_scores(val_loader, model, config["device"], criterion)

        if early_stopping is not None:
            early_stopping(mean_val_loss)
            if early_stopping.early_stop:
                break

        if scheduler is not None:
            scheduler.step(mean_val_loss)

        val_epoch_loss.append(mean_val_loss)
        train_epoch_loss.append(mean_train_loss)

        # print examples to a folder
        #save_preds_as_imgs(
        #    val_loader, model, folder="/home/feliciaj/data/Kvasir-SEG/doubleunet/", device=config["device"]
        #)
    
    loss_plot_name = "loss_" + config["model_name"]
    plt.figure(figsize=(10, 7))
    plt.plot(train_epoch_loss, color="blue", label="train loss")
    plt.plot(val_epoch_loss, color="green", label="validataion loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title(config["model_name"] + "")
    plt.legend()
    #plt.savefig(f"/home/feliciaj/PolypSegmentation/loss_plots/{loss_plot_name}.png")
        

if __name__ == "__main__":
    run_model()