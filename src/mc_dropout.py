import os
import torch
import torch.nn as nn
import torchvision
import torch.optim as optim
from torch.autograd import Variable
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
from tqdm import tqdm
import seaborn as sns

# import models
from unet import UNet_dropout, UNet
from resunetplusplus import ResUnetPlusPlus, ResUnetPlusPlus_dropout
from trainer import train_validate

# local imports
from utils.dataloader import data_loaders
from utils.utils import (
    check_scores, 
    save_grid, 
    standard_transforms
)
from utils.metrics import (
    dice_coef, 
    iou_score, 
    DiceLoss
)


class MCD(nn.Module):
    def __init__(self, model, forward_passes, device, load_folder):
        super().__init__()
        self.model = model.to(device)
        self.forward_passes = forward_passes
        self.device = device
        self.load_folder = load_folder

    def enable_dropout(self, model):
        """
        Function to enable the dropout layers during test-time
        """
        for m in model.modules():
            if m.__class__.__name__.startswith("Dropout"):
                m.train()    
    
    def forward(self, x):
        inputs = []
        for passes in range(1, self.forward_passes+1):
            checkpoint = torch.load(self.load_folder, map_location=self.device)
            self.model.load_state_dict(checkpoint["state_dict"])
            self.model.eval()
            self.enable_dropout(self.model)
            for param in self.model.parameters():
                param.requires_grad_(False)
            inputs.append(self.model(x))
        outputs = torch.stack(inputs)
        sigmoided = torch.sigmoid(outputs)
        mean = torch.mean(sigmoided, dim=0)
        variance = torch.var(sigmoided, dim=0).double()
        normalized_variance = (variance - torch.min(variance)) / (torch.max(variance) - torch.min(variance))

        return mean, normalized_variance


def save_images(batch, input, mean, variance, truth, img_folder, grid=False):
    """
    Function saves images from the same batch in a grid.
    """
    if grid==True:
        rows = 5
        save_grid(
            variance.permute(0, 2, 3, 1),
            f"{img_folder}/heatmaps/heatmap_{batch}.png",
            rows=5,
            cols=5,
        )
    else:
        rows = 1
        plt.imsave(f"{img_folder}/heatmaps/heatmap_{batch}.png", variance, cmap="turbo")

    torchvision.utils.save_image(mean, f"{img_folder}/preds/pred_{batch}.png", nrow=rows)
    torchvision.utils.save_image(truth, f"{img_folder}/masks/mask_{batch}.png", nrow=rows)
    torchvision.utils.save_image(input, f"{img_folder}/inputs/input_{batch}.png", nrow=rows)


def test_MC_dropout(model, forward_passes, loader, device, load_folder, img_folder):
    dice = 0
    for batch, (x, y) in enumerate(loader):
        x = x.to(device)
        y = y.to(device).unsqueeze(1)
        ensemble = MCD(model, forward_passes, device, load_folder)
        prob, var = ensemble(x)
        var = var.cpu().detach()
        pred = (prob > 0.5).float()
        dice += dice_coef(pred, y)
        save_images(batch, x, pred, var, y, img_folder)
    mean_dice = dice/len(test_loader)  
    print(f"Dice={mean_dice} for N={forward_passes} forward passes")


def plot_models_vs_dice(model, forward_passes, loader, device, load_folder, save_plot_path, model_name):

    dice = []
    for passes in range(1, forward_passes+1):
        running_dice = 0
        for batch, (x, y) in enumerate(loader):
            x = x.to(device)
            y = y.to(device).unsqueeze(1)
            ensemble = MCD(model, forward_passes, device, load_folder)
            prob, var = ensemble(x)
            var = var.cpu().detach()
            pred = (prob > 0.5).float()
            running_dice += dice_coef(pred, y)
        dice.append(running_dice/len(loader))

    print(dice)
    plt.figure(figsize=(8, 7))
    plt.plot(range(1, forward_passes + 1), dice, ".-", label="Dice coeff")
    plt.legend(loc="best")
    plt.grid(ls="dashed", alpha=0.7)
    plt.xticks(range(1,  forward_passes + 1))
    plt.xlabel("Number of forward passes")
    plt.ylabel("Score")
    plt.title(
        f"MC droput with {forward_passes} number of {model_name} on Kvasir-SEG"
    )
    plt.savefig(save_plot_path + f"{model_name}_dropout_{forward_passes}_models.png")


if __name__ == "__main__":
    save_path = "/home/feliciaj/PolypSegmentation/saved_models/unet_dropout/"
    save_plot_path = "/home/feliciaj/PolypSegmentation/results/plots/MC_dropout/"
    max_epoch = 150
    rates = [0, 0.1, 0.2]

    train_transforms = A.Compose(
        [
            A.Resize(height=256, width=256),
            A.RGBShift(r_shift_limit=25, g_shift_limit=25, b_shift_limit=25, p=0.9),
            A.Normalize(
                mean=[0.5579, 0.3214, 0.2350],
                std=[0.3185, 0.2218, 0.1875],
                max_pixel_value=255.0,
            ),
            A.OneOf(
                [
                    A.Blur(blur_limit=3, p=0.5),
                    A.ColorJitter(p=0.5),
                ],
                p=1.0,
            ),
            A.OneOf(
                [A.HorizontalFlip(p=1), A.RandomRotate90(p=1), A.VerticalFlip(p=1)], p=1
            ),
            ToTensorV2(),
        ]
    )

    loaders = data_loaders(
        batch_size=1,
        train_transforms=train_transforms,
        val_transforms=standard_transforms(256, 256),
        num_workers=4,
        pin_memory=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    passes = 16
    img_folder = (
        "/home/feliciaj/PolypSegmentation/results/mc_dropout_resunet++_BCE"
    )
    load_folder = (
        "/home/feliciaj/PolypSegmentation/saved_models/resunet++_dropout_BCE/resunet++_dropout_1.pt"
    )

    _, _, test_loader = loaders
    forward_passes = 16
    model = ResUnetPlusPlus_dropout(3,1).to(device) #UNet_dropout(3, 1).to(device)
    model_name = "Resunet++"

    test_MC_dropout(model, forward_passes, test_loader, device, load_folder, img_folder)

    """
    plot_models_vs_dice(
        model, 
        forward_passes, 
        test_loader, 
        device, 
        load_folder, 
        save_plot_path, 
        model_name,
    )
    """