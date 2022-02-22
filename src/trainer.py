import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Local imports
from unet_vajira import UNet_dropout
from unet import UNet  # , UNet_dropout
from resunetplusplus import ResUnetPlusPlus
from doubleunet import DoubleUNet
from utils.dataloader import data_loaders
from utils.utils import (
    check_scores,
    load_checkpoint,
    save_checkpoint,
    save_preds_as_imgs,
    EarlyStopping,
    standard_transforms,
)

from utils.metrics import BCEDiceLoss, DiceLoss


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

    tqdm_loader = tqdm(loader)  # make progress bar
    scaler = torch.cuda.amp.GradScaler()

    model.train()

    losses = []
    for batch_idx, (data, targets) in enumerate(tqdm_loader):
        # move data and masks to same device as computed gradients
        data = data.to(device=device)
        targets = (
            targets.float().unsqueeze(1).to(device=device)
        )  # add on channel dimension of 1

        with torch.cuda.amp.autocast():
            output = model(data)
            loss = criterion(output, targets)

        # update loss
        losses.append(loss.item())

        # backprop
        optimizer.zero_grad()  # zero out previous gradients
        scaler.scale(loss).backward()  # scale loss before backprop
        scaler.step(optimizer)  # update gradients
        scaler.update()  # update scale factor

        # update tqdm loop
        tqdm_loader.set_postfix(loss=loss.item())

    return sum(losses) / len(loader)


def train_validate(
    no_models,
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
):
    train_loader, val_loader, _ = loaders
    # train several models with same model architecture sequentially
    for model_idx in range(no_models):
        # zero out loss for each model
        val_epoch_loss = []
        train_epoch_loss = []
        for epoch in range(epochs):

            # train on training data
            mean_train_loss = train_model(
                train_loader, model, device, optimizer, criterion
            )
            train_epoch_loss.append(mean_train_loss)

            # check validation loss, and print validation metrics
            print("------------")
            print("At epoch %d :" % epoch)
            mean_val_loss, dice, iou = check_scores(
                val_loader, model, device, criterion
            )
            val_epoch_loss.append(mean_val_loss)

            # take scheduler step
            if scheduler is not None:
                scheduler.step(mean_val_loss)

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
                    # change name of file and run in order to save more models
                    save_checkpoint(
                        epoch,
                        checkpoint,
                        save_folder + model_name + f"_{model_idx}.pt",
                    )

            if early_stopping is not None:
                early_stopping(mean_val_loss)
                if early_stopping.early_stop:
                    break

            # save examples to a folder
            save_preds_as_imgs(
                val_loader,
                model,
                folder="/home/feliciaj/data/Kvasir-SEG/" + "/" + model_name,
                device=device,
            )

        if plot_loss:
            loss_plot_name = "loss_" + model_name + f"_{model_idx}"
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


def run_model():
    config = dict()
    config["lr"] = 1e-4
    config["device"] = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config["load_model"] = False
    config["plot_loss"] = False
    config["num_epochs"] = 150
    config["in_channels"] = 3
    config["numcl"] = 1  # no of classes/output channels
    config["batch_size"] = 32
    config["pin_memory"] = True
    config["num_workers"] = 4
    config["image_height"] = 256
    config["image_width"] = 256
    config["num_models"] = 15  # no. of models to train at once
    config["model_name"] = "resunet++"
    config["save_folder"] = (
        "/home/feliciaj/PolypSegmentation/saved_models/" + config["model_name"] + "_BCE/"
    )

    if config["model_name"] == "unet":
        model = UNet(config["in_channels"], config["numcl"]).to(config["device"])
    elif config["model_name"] == "doubleunet":
        model = DoubleUNet(config["in_channels"], config["numcl"]).to(config["device"])
    elif config["model_name"] == "resunet++":
        model = ResUnetPlusPlus(config["in_channels"], config["numcl"]).to(
            config["device"]
        )
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

    criterion = nn.BCEWithLogitsLoss() #  Sigmoid layer and the BCELoss
    #criterion = BCEDiceLoss() # Sigmoid layer and Dice + BCE loss
    #criterion = DiceLoss()  # Sigmoid layer and Dice loss

    optimizer = optim.Adam(model.parameters(), lr=config["lr"])

    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.1, patience=20, min_lr=1e-6
    )

    early_stopping = None  # EarlyStopping()

    loaders = data_loaders(
        batch_size=config["batch_size"],
        train_transforms=train_transforms,
        val_transforms=val_transforms,
        num_workers=config["num_workers"],
        pin_memory=config["pin_memory"],
    )


    train_validate(
    no_models=config["num_models"],
    epochs=config["num_epochs"],
    device=config["device"],
    criterion=criterion,
    model=model,
    optimizer=optimizer,
    scheduler=scheduler,
    loaders=loaders,
    save_folder=config["save_folder"],
    model_name=config["model_name"],
    early_stopping=early_stopping,
    plot_loss=config["plot_loss"],
    )


if __name__ == "__main__":
    run_model()
