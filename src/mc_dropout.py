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


class UNetClassifier():
    def __init__(self, device, droprate=0.3, max_epoch=150, lr=0.01):
        self.device = device
        self.max_epoch = max_epoch
        self.lr = lr
        self.model = UNet_dropout(in_channels=3, out_channels=1, droprate=droprate)
        self.model.to(device)
        self.criterion = DiceLoss().to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, factor=0.1, patience=20, min_lr=1e-6) 
        self.loss_ = []
        self.val_dice = []
        self.val_iou = []
        
    def train_model(self, trainloader, valloader, verbose=True):
        device = self.device

        X_test, y_test = iter(valloader).next()
        X_test = X_test.to(device)

        tqdm_loader = tqdm(trainloader) # make progress bar
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

            mean_val_loss, mean_val_dice, mean_val_iou = check_scores(valloader, self.model, self.device, self.criterion)
            self.val_dice.append(mean_val_dice)
            self.val_iou.append(mean_val_iou)
            self.loss_.append(mean_val_loss)

            if self.scheduler is not None:
                    self.scheduler.step(mean_val_loss)
            
            
            if verbose:
                print("Epoch {} ==> loss: {}".format(epoch+1, self.loss_[-1]))
                print("---------")

        return self.val_dice
    
    def test_model(self, test_loader, forward_passes): 
        model = self.model.eval()
        enable_dropout(model)
        dice = 0
        with torch.no_grad():
            for x, y in test_loader:
                x = x.to(device=device)
                y = y.to(device=device).unsqueeze(1)
                prob = torch.sigmoid(model(x))
                pred = (prob > 0.5).float()
                dice += dice_coef(pred, y)

        model = self.model.train()
        return pred, variance, dice/len(test_loader)
    


def enable_dropout(model):
    """ 
    Function to enable the dropout layers during test-time 
    """
    for m in model.modules():
        if m.__class__.__name__.startswith("Dropout"):
            m.train()


def test_MC_model(loader, forward_passes, model, device, save_folder):
    """ 
    Function to get the monte-carlo samples and uncertainty estimates
    through multiple forward passes

    Parameters
    ----------
    loader : object
        data loader object from the data loader module
    forward passes : int
        number of monte-carlo models
    model : object
        pytorch model
    device : str
        cuda object
    """

    dice, iou = 0, 0
    for batch, (x, y) in enumerate(loader):
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
        mean = torch.mean(outputs, dim=0).double()
        variance = torch.mean((outputs**2 - mean), dim=0).double().cpu()

        mean_pred = (mean > 0.5).float() 
        dice += dice_coef(mean_pred, y)
        iou += iou_score(mean_pred, y)
        
        torchvision.utils.save_image(mean_pred, f"{save_folder}/pred_{batch}.png")
        torchvision.utils.save_image(y, f"{save_folder}/mask_{batch}.png")
        save_grid(variance.permute(0,2,3,1), f"{save_folder}/heatmap_{batch}.png", rows=4, cols=8)

    print(f"IoU score: {iou/len(loader)}")
    print(f"Dice score: {dice/len(loader)}")

    return dice/len(loader), iou/len(loader)


def plot_dropout_vs_forward_passes(forward_passes, save_plot_path, load_model_path):

    train_loader, val_loader, test_loader = data_loaders(
        batch_size=32, 
        train_transforms=None, 
        val_transforms=standard_transforms(256,256), 
        num_workers=4, 
        pin_memory=True
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load(load_model_path)
    save_folder = "/home/feliciaj/PolypSegmentation/results/mc_dropout/"
    dice_list, iou_list = [], []
    for passes in range(1, forward_passes+1):
        dice, iou = test_MC_model(test_loader, passes, model, device, save_folder)
        dice_list.append(dice)
        iou_list.append(iou)

    plt.figure(figsize=(8,7))
    plt.plot(range(1,forward_passes+1), dice_list, ".-", label="Dice coeff")
    plt.plot(range(1,forward_passes+1), iou_list, ".-", label="IoU")
    plt.legend(loc="best");
    plt.xlabel("Number of networks");
    plt.ylabel("Score");
    plt.title(f"Dice and IoU scores with MC droput of droprate=0.1 with {forward_passes} number of U-Nets")
    plt.savefig(save_plot_path+f"unet_dropout_{forward_passes}_models.png")


def plot_dropout_models(max_epoch, loaders, save_path: str, save_plot_path: str, train=False, rates = [0, 0.3, 0.5]):
    train_loader, val_loader, test_loader = loaders

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

    unets = []
    for rate in rates:
        unets.append(UNetClassifier(device=device, droprate=rate, max_epoch=max_epoch))

    if train:        
        # Training, set verbose=True to see loss after each epoch
        [unet.train_model(train_loader, val_loader, verbose=True) for unet in unets]

        # Save trained models
        for idx, (rate, unet) in enumerate(zip(rates, unets)):
            torch.save(unet.model, save_path+f"unet_{idx}.pt")
            torch.save(unet.val_dice, save_path+f"unet_val_dices_{idx}.pt")
            

    plt.figure(figsize=(8,7))
    for idx, (rate, unet) in enumerate(zip(rates, unets)):
        #model = torch.load(save_path+f"unet_{idx}.pt")
        dices = torch.load(save_path+f"unet_val_dices_{idx}.pt", map_location=torch.device("cpu"))
        if rate==0:
            label="U-Net no dropout"
        else:
            label = f"U-Net dropout rate={rate:.1f}"
        plt.plot(range(1,len(dices)+1), dices, ".-", label=label)

    plt.legend(loc="best");
    plt.xlabel("Epochs");
    plt.ylabel("Dice coefficient");
    plt.title("Dice coefficient scores from Kvasir-SEG Dataset for all Networks")
    plt.savefig(save_plot_path+"unet.png")


if __name__ == "__main__":
    save_path = "/home/feliciaj/PolypSegmentation/saved_models/unet_dropout/"
    save_plot_path = "/home/feliciaj/PolypSegmentation/results/figures/"
    max_epoch = 150
    rates = [0, 0.1, 0.2]
       
    train_transforms = A.Compose([
        A.Resize(height=256, width=256),
        A.RGBShift(r_shift_limit=25, g_shift_limit=25, b_shift_limit=25, p=0.9),
        A.Normalize(
                mean=[0.5579, 0.3214, 0.2350],
                std=[0.3185, 0.2218, 0.1875],
                max_pixel_value=255.0,
        ),
        A.OneOf([
            A.Blur(blur_limit=3, p=0.5),
            A.ColorJitter(p=0.5),
        ], p=1.0),
        A.OneOf([
                A.HorizontalFlip(p=1),
                A.RandomRotate90(p=1),
                A.VerticalFlip(p=1)            
        ], p=1),
        ToTensorV2(),
    ])

    loaders = data_loaders(
        batch_size=32, 
        train_transforms=train_transforms, 
        val_transforms=standard_transforms(256,256), 
        num_workers=4, 
        pin_memory=True
    )

    #plot_dropout_models(max_epoch, loaders, save_path, save_plot_path, False, rates)

    train_loader, val_loader, test_loader = loaders
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load(save_path+"unet_1.pt")
    save_folder = "/home/feliciaj/PolypSegmentation/results/mc_dropout/"
    test_MC_model(test_loader, 30, model, device, save_folder)
    load_model_path = "/home/feliciaj/PolypSegmentation/saved_models/unet_dropout/unet_1.pt"
    plot_dropout_vs_forward_passes(50, save_plot_path, load_model_path)