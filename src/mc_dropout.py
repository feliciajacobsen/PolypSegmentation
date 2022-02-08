import os
import torch
import torch.nn as nn
import torchvision
import torch.optim as optim
from torch.autograd import Variable
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# import models
from resunetplusplus import ResUnetPlusPlus
from doubleunet import DoubleUNet
from unet import UNet_dropout, UNet

# import from utils subfolder
from utils.dataloader import data_loaders
from utils.utils import check_scores, save_grid, standard_transforms
from utils.metrics import dice_coef, iou_score, DiceLoss


class MCDropoutSegmentation:
    def __init__(self, device, loaders, droprate=0.3, max_epoch=150, lr=0.01):
        self.device = device
        self.max_epoch = max_epoch
        self.lr = lr
        self.train_loader, self.val_loader, self.test_loader = loaders
        self.model = UNet_dropout(in_channels=3, out_channels=1, droprate=droprate)
        self.model.to(device)
        self.criterion = DiceLoss().to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, factor=0.1, patience=20, min_lr=1e-6
        )
        self.loss_ = []
        self.val_dice = []
        self.val_iou = []

    def train_model(self, verbose=True):
        """
        Train model and stores validation dice coefficients for 
        each epoch.

        Parameters
        ----------
        train_loader : object
            data loader object from the data loader module
        val_loader : object
            data loader object from the data loader module
        
        Returns:
        ----------
        val_dice : list
            list of dice coefficients stored and updated at each epoch
        """

        X_test, y_test = iter(val_loader).next()
        X_test = X_test.to(self.device)

        tqdm_loader = tqdm(self.train_loader)  # make progress bar
        scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.max_epoch):
            for i, data in enumerate(tqdm_loader):
                targets, labels = data
                targets, labels = Variable(targets).cuda(), Variable(labels).cuda()

                with torch.cuda.amp.autocast():
                    outputs = self.model(targets)
                    loss = self.criterion(outputs, labels)

                self.optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                scaler.update()
                tqdm_loader.set_postfix(loss=loss.item())

            mean_val_loss, mean_val_dice, mean_val_iou = check_scores(
                self.val_loader, self.model, self.device, self.criterion
            )
            self.val_dice.append(mean_val_dice)
            self.val_iou.append(mean_val_iou)
            self.loss_.append(mean_val_loss)

            if self.scheduler is not None:
                self.scheduler.step(mean_val_loss)

            if verbose:
                print("Epoch {} ==> loss: {}".format(epoch + 1, self.loss_[-1]))
                print("---------")

        return self.val_dice

    def train_n_models(self, save_path, save_plot_path, fig_name, rates=[0, 0.3, 0.5], plot=False):
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
            unets.append(MCDropoutSegmentation(self.device, rate, self.max_epoch))

        # Training, set verbose=True to see loss after each epoch
        [unet.train_model(train_loader, val_loader, verbose=True) for unet in unets]

        # Save trained models
        for idx, (rate, unet) in enumerate(zip(rates, unets)):
            torch.save(unet.model, save_path + f"unet_{idx}.pt")
            torch.save(unet.val_dice, save_path + f"unet_val_dices_{idx}.pt")

        if plot:
            plt.figure(figsize=(8, 7))
            for idx, (rate, unet) in enumerate(zip(rates, unets)):
                dices = torch.load(
                    save_path + f"unet_val_dices_{idx}.pt", map_location=torch.device("cpu")
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
             
    def enable_dropout(self, model):
        """
        Function to enable the dropout layers during test-time
        """
        for m in model.modules():
            if m.__class__.__name__.startswith("Dropout"):
                m.train()

    def save_prediction_imgs(self, forward_passes, save_img_folder):
        """
        Function to get the monte-carlo samples and uncertainty estimates
        through multiple forward passes

        Parameters
        ----------
        forward passes : int
            number of monte-carlo models
        save_img_folder : str
            path to store prediction images

        Returns
        ----------
            None  
        """
        device = self.device
        model = torch.load(self.save_path)

        dice, iou = 0, 0
        for batch, (x, y) in enumerate(self.test_loader):
            preds = []
            x = x.to(device)
            y = y.to(device).unsqueeze(1)
            # set non-dropout layers to eval mode
            model.eval()
            # set dropout layers to train mode
            enable_dropout(model)
            for i in range(forward_passes):
                with torch.no_grad():
                    output = model(x)
                    prob = torch.sigmoid(output)
                    preds.append(prob)
            outputs = torch.stack(preds)
            # double precision for mean and variance tensors
            mean = torch.mean(outputs, dim=0).double()
            variance = torch.mean((outputs**2 - mean), dim=0).double().cpu()

            mean_pred = (mean > 0.5).float()
            dice += dice_coef(mean_pred, y)
            iou += iou_score(mean_pred, y)

            torchvision.utils.save_image(mean_pred, f"{save_folder}/pred_{batch}.png")
            torchvision.utils.save_image(y, f"{save_folder}/mask_{batch}.png")
            save_grid(
                variance.permute(0, 2, 3, 1),
                f"{save_folder}/heatmap_{batch}.png",
                rows=4,
                cols=8,
            )

        # mean scores 
        print(f"IoU score: {iou/len(loader)}")
        print(f"Dice score: {dice/len(loader)}")

    
    def save_scores(self, model_load_path,forward_passes):
        model = torch.load(model_load_path)
        device = self.device
        test_loader = self.test_loader

        self.dice_list, self.iou_list = [], []
        model.eval()
        enable_dropout(model)
        for passes in range(1, forward_passes + 1):
            dice, iou = 0, 0
            for x, y in test_loader:
                with torch.no_grad():
                    x = x.to(device)
                    y = y.to(device)
                    if y.shape[1] != 1:
                        y = y.unsqueeze(1)
                    output = model(x)
                    prob = torch.sigmoid(output)
                    pred = (pred > 0.5).float()

                    dice += dice_coef(pred, y)
                    iou += iou_score(pred, y)
            self.dice_list.append(dice/len(test_loader))  
            self.iou_list.append(iou/len(test_loader))  

        return self.dice_list, self.iou_list
        


def plot_dropout_vs_forward_passes(dice_list, iou_list, save_plot_path):
    assert (len(dice_list) == len(iou_list)), "Error: Dice coeff list does not match IoU score list!"

    forward_passes = len(dice_list)
    plt.figure(figsize=(8, 7))
    plt.plot(range(1, forward_passes + 1), dice_list, ".-", label="Dice coeff")
    plt.plot(range(1, forward_passes + 1), iou_list, ".-", label="IoU")
    plt.legend(loc="best")
    plt.xlabel("Number of networks")
    plt.ylabel("Score")
    plt.title(
        f"Dice and IoU scores with MC droput of droprate=0.1 with {forward_passes} number of U-Nets"
    )
    plt.savefig(save_plot_path + f"unet_dropout_n={forward_passes}_models.png")



if __name__ == "__main__":
    save_path = "/home/feliciaj/PolypSegmentation/saved_models/unet_dropout/"
    save_plot_path = "/home/feliciaj/PolypSegmentation/results/figures/"
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
        batch_size=32,
        train_transforms=train_transforms,
        val_transforms=standard_transforms(256, 256),
        num_workers=4,
        pin_memory=True,
    )

    # plot_dropout_models(max_epoch, train_loader, val_loader, save_path, save_plot_path, False, rates)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clf = MCDropoutSegmentation(device, loaders)
    model_path = "/home/feliciaj/PolypSegmentation/saved_models/unet_dropout/unet_1.pt" # path to where model is stored
    dice_list, iou_list = clf.save_scores(model_path, 20)

    save_plot_path = "/home/feliciaj/PolypSegmentation/results/figures"
    plot_dropout_vs_forward_passes(dice_list, iou_list, save_plot_path)