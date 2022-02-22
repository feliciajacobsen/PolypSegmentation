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
#from unet import UNet_dropout, UNet
from unet_vajira import UNet_dropout
# import from utils subfolder
from utils.dataloader import data_loaders
from utils.utils import check_scores, save_grid, standard_transforms
from utils.metrics import dice_coef, iou_score, DiceLoss


class MCDropoutSegmentation:
    def __init__(self, device, loaders, droprate=0.3, lr=0.01):
        self.device = device
        self.lr = lr
        self.loaders = loaders
        self.train_loader, self.val_loader, self.test_loader = loaders
        self.model = UNet_dropout(in_channels=3, out_channels=1)#UNet_dropout(in_channels=3, out_channels=1, droprate=droprate)
        self.model.to(device)
        self.criterion = DiceLoss().to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, factor=0.1, patience=20, min_lr=1e-6
        )
        self.loss_ = []
        self.val_dice = []
        self.val_iou = []

    def train_model(self, save_model_path, max_epoch=150, verbose=True):
        """
        Train model and stores validation dice coefficients for
        each epoch.

        Parameters
        ----------
        save_model_path : str
            path and filename to where to store model

        Returns:
        ----------
        val_dice : list
            list of dice coefficients stored and updated at each epoch
        """

        X_test, y_test = iter(self.val_loader).next()
        X_test = X_test.to(self.device)

        tqdm_loader = tqdm(self.train_loader)  # make progress bar
        scaler = torch.cuda.amp.GradScaler()

        model.train()
        for epoch in range(max_epoch):
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

            if epoch == max_epoch - 1:
                torch.save(self.model, save_model_path + ".pt")

            if verbose:
                print("Epoch {} ==> loss: {}".format(epoch + 1, self.loss_[-1]))
                print("---------")

        return self.val_dice

    def train_n_models(
        self, save_path, save_plot_path, fig_name, rates=[0, 0.1 ,0.3, 0.5], plot=False
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
                    device=self.device, 
                    loaders=self.loaders, 
                    droprate=rate
                )
            )

        # Save trained models
        for idx, (rate, unet) in enumerate(zip(rates, unets)):
            unet.train_model(save_path + f"unet_rate={rate}.pt", verbose=True)
            #torch.save(unet.model, save_path + f"unet_rate={rate}.pt")
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

    def enable_dropout(self, model):
        """
        Function to enable the dropout layers during test-time
        """
        for m in model.modules():
            if m.__class__.__name__.startswith("Dropout"):
                m.train()

    def predict(self, forward_passes, load_path, save_imgs=False, img_folder=None):
        """
        Function to get the monte-carlo samples and uncertainty estimates
        through multiple forward passes

        Parameters
        ----------
        forward passes : int
            number of monte-carlo models
        load_folder : str
            path to where trained images are stores

        Returns
        ----------
        mean: tensor
            Mean image of forward passes
        variance : tensor
            Variance of the forward passs.
        """
        device = self.device
        model = torch.load(load_path)

        for batch, (x, y) in enumerate(self.test_loader):
            preds = []
            x = x.to(device)
            y = y.to(device).unsqueeze(1)

            # set non-dropout layers to eval mode
            model.eval()

            # set dropout layers to train mode
            self.enable_dropout(model)

            for passes in range(forward_passes):
                with torch.no_grad():
                    output = model(x)
                    prob = torch.sigmoid(output)
                    preds.append(prob)
            outputs = torch.stack(preds) # shape – (forward_passes, b, c, h, w)

            # double precision for mean and variance tensors
            mean = torch.mean(outputs, dim=0).double()
            #variance = torch.mean((outputs - mean)**2, dim=0).double().cpu()
            variance = torch.var(outputs, dim=0).double().cpu()

            # calculating mean prediction across multiple MCD forward passes
            mean_pred = (mean > 0.5).float()

            if save_imgs==True:
                self.save_images(batch, mean_pred, variance, y, img_folder)

        model.train()

        return mean_pred, variance

    def save_images(self, batch, mean, variance, truth, img_folder):
        """
        Function saves images from the same batch in a grid. 
        """   
        torchvision.utils.save_image(mean, f"{img_folder}/pred_{batch}.png")
        torchvision.utils.save_image(truth, f"{img_folder}/mask_{batch}.png")
        save_grid(
            variance.permute(0, 2, 3, 1),
            f"{img_folder}/heatmap_{batch}.png",
            rows=4,
            cols=8,
        )
        

    def save_scores(self, model_load_path, forward_passes):
        model = torch.load(model_load_path)
        device = self.device
        test_loader = self.test_loader

        self.dice_list, self.iou_list = [], []
        for passes in range(1, forward_passes + 1):
            dice, iou = 0, 0
            mean, variance = self.predict(passes)

            for x, y in test_loader:
                x = x.to(device)
                y = y.to(device)
                if y.shape[1] != 1:
                    y = y.unsqueeze(1)    
                dice += dice_coef(mean, y)
                iou += iou_score(mean, y)

            self.dice_list.append(dice / len(test_loader))
            self.iou_list.append(iou / len(test_loader))

        return self.dice_list, self.iou_list


def plot_dropout_vs_forward_passes(dice_list, iou_list, save_plot_path):
    assert len(dice_list) == len(
        iou_list
    ), "Error: Dice coeff list does not match IoU score list!"

    forward_passes = len(dice_list)
    plt.figure(figsize=(8, 7))
    plt.plot(range(1, forward_passes + 1), dice_list, ".-", label="Dice coeff")
    plt.plot(range(1, forward_passes + 1), iou_list, ".-", label="IoU")
    plt.legend(loc="best")
    plt.xlabel("Number of forward passes")
    plt.ylabel("Score")
    plt.title(
        f"Dice and IoU scores with MC droput with {forward_passes} number of U-Nets"
    )
    plt.savefig(save_plot_path + f"unet_dropout_{forward_passes}_models.png")


if __name__ == "__main__":
    save_path = "/home/feliciaj/PolypSegmentation/saved_models/unet_dropout_vajira/"
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    obj = MCDropoutSegmentation(device, loaders, droprate=0.3, max_epoch=10, lr=0.01)

    passes = 5
    img_folder = "/home/feliciaj/PolypSegmentation/results/mc_dropout_unet"
    load_path= "/home/feliciaj/PolypSegmentation/saved_models/unet_dropout/unet_rate=0.1.pt"
    mean, var = obj.predict(passes,load_path,True,img_folder)
    

    save_path = "/home/feliciaj/PolypSegmentation/saved_models/unet_dropout/"
    save_plot_path = "/home/feliciaj/PolypSegmentation/results/figures/"
    rates = [0, 0.1 ,0.3, 0.5]
    fig_name = f"U-Net with dropout predicted on Kvasir-SEG validation data with rates={rates}"
    #obj.train_n_models(save_path, save_plot_path, fig_name, rates, True) # train model

    #obj.train_model(save_model_path="/home/feliciaj/PolypSegmentation/saved_models/vajira/unet")
    
    #model_path = "/home/feliciaj/PolypSegmentation/saved_models/vajira/unet.pt"  # path to where model is stored
    #dice_list, iou_list = obj.save_scores(model_path, 15)

    #save_plot_path = "/home/feliciaj/PolypSegmentation/results/figures/"
    #plot_dropout_vs_forward_passes(dice_list, iou_list, save_plot_path)
    