import sys
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Local imports
from unet import UNet, UNet_dropout
from resunetplusplus import ResUnetPlusPlus, ResUnetPlusPlus_dropout
from utils.dataloader import data_loaders
from utils.utils import (
    check_scores,
    load_checkpoint,
    save_checkpoint,
    save_preds_as_imgs,
    EarlyStopping,
    standard_transforms,
)

from utils.metrics import BCEDiceLoss, DiceLoss, dice_coef, iou_score


def train_model(loader, model, device, optimizer, criterion, scheduler, epoch):
    """
    Function perform one epoch on entire dataset and prints loss for each batch.

    Args:
        loader (object): iterable-style dataset.
        model (class): provides with a forward method.
        device (cuda object): cpu or gpu.
        optimizer (torch object): optimization algorithm.
        criterion (torch object): loss function with backward method.
        scheduler (torch object): learning rate scheduler.

    Returns:
        Mean loss
    """

    tqdm_loader = tqdm(loader)  # make progress bar
    scaler = torch.cuda.amp.GradScaler() # set scaler

    model.train()
    losses = []
    for batch_idx, (data, targets) in enumerate(tqdm_loader):
        # move data and masks to same device as computed gradients
        data = data.to(device=device)
        # add channel dimension
        targets = targets.unsqueeze(1).to(device=device)

        # forward
        with torch.cuda.amp.autocast():
            output = model(data)
            loss = criterion(output, targets)

        # update loss
        losses.append(loss.item())

        # backprop
        optimizer.zero_grad()  # zero out previous gradients
        scaler.scale(loss).backward()  # scale loss in backprop
        scaler.step(optimizer)  # update gradients
        scaler.update()  # update scale factor

        # update tqdm loop
        tqdm_loader.set_postfix(loss=loss.item())

    mean_loss = sum(losses) / len(loader)

    # take scheduler step
    if scheduler is not None:
        #scheduler.step(mean_loss)
        scheduler.step(epoch=epoch)

    return mean_loss


def train_validate(
    epochs,
    device,
    criterion,
    model,
    optimizer,
    scheduler,
    loaders,
    save_folder,
    model_name,
    early_stopping,
    plot_loss,
    number,
):
    train_loader, val_loader, _ = loaders
    val_epoch_loss, train_epoch_loss = [], []
    dice = []

    for epoch in range(epochs):
        # train on training data
        mean_train_loss = train_model(train_loader, model, device, optimizer, criterion, scheduler,epoch)
        train_epoch_loss.append(mean_train_loss)

        # check validation loss, and print validation metrics
        print("------------")
        print("At epoch %d :" % epoch)
        mean_val_loss, running_dice, _ = check_scores(val_loader, model, device, criterion)
        val_epoch_loss.append(mean_val_loss)
        dice.append(running_dice)

        # save model after training
        if epoch == epochs - 1:
            if save_folder != None:
                checkpoint = {
                    "epoch": epoch,
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "criterion": criterion.state_dict(),
                    "loss": mean_val_loss,
                }
                
                save_checkpoint(
                    epoch,
                    checkpoint,
                    save_folder + model_name + f"_{number}.pt",
                )
        
        if early_stopping is not None:
            early_stopping(mean_val_loss)
            if early_stopping.early_stop:
                break

        # save examples to a folder
        save_preds_as_imgs(
            val_loader,
            model,
            folder="/home/feliciaj/data/Kvasir-SEG/" + model_name,
            device=device,
        )

    if plot_loss:
        loss_plot_name = "loss_" + model_name
        plt.figure(figsize=(10, 7))
        plt.plot(train_epoch_loss, color="blue", label="train loss")
        plt.plot(val_epoch_loss, color="green", label="validataion loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title(model_name)
        plt.legend()
        plt.savefig(
            f"/home/feliciaj/PolypSegmentation/results/loss_plots/{loss_plot_name}.png"
        )


def run_model(number):
    config = dict()
    config["lr"] = 1e-4
    config["device"] = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config["plot_loss"] = False # set to true if we want to plot loss
    config["early_stopping"] = None
    config["num_epochs"] = 90
    config["in_channels"] = 3
    config["numcl"] = 1  # no of classes/output channels
    config["batch_size"] = 32
    config["pin_memory"] = True
    config["num_workers"] = 4
    config["image_height"] = 256
    config["image_width"] = 256
    config["model_name"] = "resunet++_dropout" 
    config["save_folder"] = (
        "/home/feliciaj/PolypSegmentation/saved_models/"
        + config["model_name"] + "/"
    )

    if config["model_name"] == "unet":
        model = UNet(config["in_channels"], config["numcl"]).to(config["device"])

    elif config["model_name"] == "resunet++":
        model = ResUnetPlusPlus(config["in_channels"], config["numcl"]).to(config["device"])

    elif config["model_name"] == "unet_dropout":
        model = UNet_dropout(config["in_channels"], config["numcl"]).to(config["device"])

    elif config["model_name"] == "resunet++_dropout":
        model = ResUnetPlusPlus_dropout(config["in_channels"], config["numcl"]).to(config["device"])    

    else:
        print("ERROR: Model not found!")

    train_transforms = A.Compose(
        [
            A.Resize(height=config["image_height"], width=config["image_width"]),
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

    val_transforms = standard_transforms(config["image_height"], config["image_width"])

    criterion = nn.BCEWithLogitsLoss()  #  Sigmoid layer and the BCELoss
    #criterion = DiceLoss()  # Sigmoid layer and Dice loss

    optimizer = optim.Adam(model.parameters(), lr=config["lr"])
    
    scheduler = None
    """
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, 
        T_0=20, 
        T_mult=1, 
        eta_min=1e-8, 
        verbose=True,
    )
    """
    
    loaders = data_loaders(
        batch_size=config["batch_size"],
        train_transforms=train_transforms,
        val_transforms=val_transforms,
        num_workers=config["num_workers"],
        pin_memory=config["pin_memory"],
    )
    
    train_validate(
        epochs=config["num_epochs"],
        device=config["device"],
        criterion=criterion,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        loaders=loaders,
        save_folder=config["save_folder"],
        model_name=config["model_name"],
        early_stopping=config["early_stopping"],
        plot_loss=config["plot_loss"],
        number=number,
    )



if __name__ == "__main__":
    number = 1 #int(sys.argv[1]) # number of models trained sequentially
    run_model(number)
