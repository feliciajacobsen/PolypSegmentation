import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from Unet import UNET
from double_Unet import double_UNET
from dataloader import data_loader

def check_accuracy(loader, model, device="cuda"):
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device).unsqueeze(1)
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)
            dice_score += (2 * (preds * y).sum()) / (
                (preds + y).sum() + 1e-8
            )

    print(
        f"Got {num_correct}/{num_pixels} with acc {num_correct/num_pixels*100:.2f}"
    )
    print(f"Dice score: {dice_score/len(loader)}")
    model.train()



def save_predictions_as_imgs(
    loader, model, folder="saved_images/", device="cuda"
):
    model.eval()
    for idx, (x, y) in enumerate(loader):
        x = x.to(device=device)
        with torch.no_grad():
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
        torchvision.utils.save_image(
            preds, f"{folder}/pred_{idx}.png"
        )
        torchvision.utils.save_image(y.unsqueeze(1), f"{folder}{idx}.png")

    model.train()


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

def run_model():
    config = dict()
    config["lr"] = 1e-4
    config["device"] = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    config["load_model"] = False
    config["num_epochs"] = 10
    config["batch_size"] = 64
    config["pin_memory"] = False

    train_transforms = A.Compose(
        [   
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
            A.Normalize(
                mean=[0.0,0.0,0.0],
                std=[1.0,1.0,1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )
   
    model = UNET(in_channels=3, out_channels=1, features=[64,128,256,512]).to(device=config["device"])
    loss_fn = nn.BCEWithLogitsLoss() #  Sigmoid layer and the BCELoss 
    optimizer = optim.Adam(model.parameters(), lr=config["lr"])

    train_loader= data_loader(0.8, config["batch_size"], "train", config["pin_memory"], train_transforms)
    val_loader = data_loader(0.8, config["batch_size"], "val", config["pin_memory"], val_transforms)


    if config["load_model"]:
        load_checkpoint(torch.load("my_checkpoint.pt"), model)

    check_accuracy(val_loader, model, device=config["device"])


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
            val_loader, model, folder="~/data/Kvasir-SEG/saved_images/", device=config["device"]
        )



if __name__ == "__main__":
    run_model()