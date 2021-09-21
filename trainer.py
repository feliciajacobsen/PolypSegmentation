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
    train_transforms = A.Compose(
        [
            A.Resize(height=config["image_height"], width=config["image_width"]),
            A.Rotate(limit=35, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.Normalize(
                mean=[0.0,0.0,0.0],
                std=[1.0,1.0,1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

    val_transforms = A.Compose(
        [
            A.Resize(height=config["image_height"], width=config["image_width"]),
            A.Normalize(
                mean=[0.0,0.0,0.0],
                std=[1.0,1.0,1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )
    """
    Here you can change the out_channels to number of classes 
    and the loss function to cross entropy loss in order 
    to extrend to a multi-class problem
    """
    model = UNET(in_channels=3, out_channels=1).to(device=config["device"])
    loss_fn = nn.BCEWithLogitsLoss() #  Sigmoid layer and the BCELoss 
    optimizer = optim.Adam(model.parameters(), lr=config["lr"])

    train_loader, val_loader = get_loaders(
        config["train_img_dir"],
        config["train_mask_dir"],
        config["val_img_dir"],
        config["val_mask_dir"],
        config["batch_size"],
        train_transforms,
        val_transforms,
        config["num_workers"],
        config["pin_memory"],
    )

    if config["load_model"]:
        load_checkpoint(torch.load("my_checkpoint.pt"), model)

    check_accuracy(val_loader, model, device=config["device"])

    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(config["num_epochs"]):
        train_fn(train_loader, model, optimizer, loss_fn, scaler)

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
            val_loader, model, folder="save_images/", device=config["device"]
        )
