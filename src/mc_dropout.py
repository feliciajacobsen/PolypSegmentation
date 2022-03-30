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

# import models
from resunetplusplus import ResUnetPlusPlus
from doubleunet import DoubleUNet
from unet import UNet_dropout, UNet
# from unet_vajira import UNet_dropout

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
        self.model = model
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


class MCDropoutSegmentation:
    def __init__(self, device, loaders, droprate=0.1, lr=0.01):
        self.device = device
        self.lr = lr
        self.loaders = loaders
        self.train_loader, self.val_loader, self.test_loader = loaders
        self.model = UNet_dropout(
            in_channels=3, out_channels=1
        ).to(device)  # UNet_dropout(in_channels=3, out_channels=1, droprate=droprate)
        self.criterion = DiceLoss().to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, factor=0.1, patience=20, min_lr=1e-6
        )
        self.loss_ = []
        self.val_dice = []
        self.val_iou = []

    def train_n_models(
        self, save_path, save_plot_path, fig_name, rates=[0, 0.1, 0.3, 0.5], plot=False
    ):
        """
        save_path : str
            path to where to save trained model and their dice scores
        save_plot_path : str
            path to where to store plot
        fig_name : str
            filename of plot
        rates : list
            list of floats
        plot : boolean
            if true, dice vs. epochs are saved.
        """

        self.save_path = save_path
        train_loader = self.train_loader
        val_loader = self.val_loader

        unets = []
        for rate in rates:
            unets.append(
                MCDropoutSegmentation(
                    device=self.device, loaders=self.loaders, droprate=rate
                )
            )

        # Save trained models
        for idx, (rate, unet) in enumerate(zip(rates, unets)):
            unet.train_model(save_path + f"unet_rate={rate}.pt", verbose=True)
            # torch.save(unet.model, save_path + f"unet_rate={rate}.pt")
            torch.save(unet.val_dice, save_path + f"unet_val_dices_rate={rate}.pt")

        if plot:
            plt.figure(figsize=(8, 7))
            for idx, (rate, unet) in enumerate(zip(rates, unets)):
                dices = torch.load(
                    save_path + f"unet_val_dices_rate={rate}.pt",
                    map_location=torch.device("cpu"),
                )
                if rate == 0:
                    label = "U-Net no dropout"
                else:
                    label = f"U-Net dropout rate={rate:.1f}"
                plt.plot(range(1, len(dices) + 1), dices, ".-", label=label)

            plt.legend(loc="best")
            plt.xlabel("Epochs")
            plt.ylabel("Dice coefficient")
            plt.title("Dice coefficient scores on Kvasir-SEG validation dataset")
            plt.savefig(save_plot_path + f"{fig_name}.png")


def save_images(batch, input, mean, variance, truth, img_folder):
    """
    Function saves images from the same batch in a grid.
    """
    torchvision.utils.save_image(mean, f"{img_folder}/pred_{batch}.png", nrow=5)
    torchvision.utils.save_image(truth, f"{img_folder}/mask_{batch}.png", nrow=5)
    torchvision.utils.save_image(input, f"{img_folder}/input_{batch}.png", nrow=5)
    save_grid(
        variance.permute(0, 2, 3, 1),
        f"{img_folder}/heatmap_{batch}.png",
        rows=5,
        cols=5,
    )


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


def plot_models_vs_dice(model, forward_passes, loader, device, load_folder, save_plot_path):

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

    plt.figure(figsize=(8, 7))
    plt.plot(range(1, forward_passes + 1), dice, ".-", label="Dice coeff")
    plt.legend(loc="best")
    plt.grid(ls="dashed", alpha=0.7)
    plt.xticks(range(1,  forward_passes + 1))
    plt.xlabel("Number of forward passes")
    plt.ylabel("Score")
    plt.title(
        f"MC droput with {forward_passes} number of U-Nets on Kvasir-SEG"
    )
    plt.savefig(save_plot_path + f"unet_dropout_{forward_passes}_models.png")


if __name__ == "__main__":
    save_path = "/home/feliciaj/PolypSegmentation/saved_models/unet_dropout_vajira/"
    save_plot_path = "/home/feliciaj/PolypSegmentation/results/results_kvasir/plots/MC_dropout/"
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
        batch_size=25,
        train_transforms=train_transforms,
        val_transforms=standard_transforms(256, 256),
        num_workers=4,
        pin_memory=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    obj = MCDropoutSegmentation(device, loaders, droprate=0.3, lr=0.01)

    passes = 5
    img_folder = (
        "/home/feliciaj/PolypSegmentation/results/results_kvasir/mc_dropout_unet_dice"
    )
    load_path = (
        "/home/feliciaj/PolypSegmentation/saved_models/unet_dropout/unet_dropout.pt"
    )

    save_path = "/home/feliciaj/PolypSegmentation/saved_models/unet_dropout/"
    rates = [0, 0.1, 0.3, 0.5]
    fig_name = (
        f"U-Net with dropout predicted on Kvasir-SEG validation data with rates={rates}"
    )
    #obj.train_n_models(save_path, save_plot_path, fig_name, rates, True) # train model
    #obj.train_model(save_model_path="/home/feliciaj/PolypSegmentation/saved_models/vajira/unet")

    _, _, test_loader = loaders
    forward_passes = 16
    model = UNet_dropout(3, 1).to(device)
    load_folder = load_path

    test_MC_dropout(model, forward_passes, test_loader, device, load_folder, img_folder)

    plot_models_vs_dice(model, forward_passes, test_loader, device, load_folder, save_plot_path)