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

from dataloader import data_loaders

from utils import check_scores, save_grid, standard_transforms
from metrics import dice_coef, iou_score, DiceLoss


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
    
    def test_model(self, x): 
        model = self.model.eval()
        with torch.no_grad():
            prob = torch.sigmoid(model(x))
            pred = (prob > 0.5).float()
        model = self.model.train()
        return pred


def enable_dropout(model):
    """ 
    Function to enable the dropout layers during test-time 
    """
    for m in model.modules():
        if m.__class__.__name__.startswith("Dropout"):
            m.train()



def get_monte_carlo_predictions(loader,forward_passes,model,n_classes,n_samples,device):
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
    n_classes : int
        number of classes in the dataset
    n_samples : int
        number of samples in the test set
    """

    dropout_predictions = np.empty((0, n_samples, n_classes))
    softmax = nn.Softmax(dim=1)
    for i in range(forward_passes):
        preds = np.empty((0, n_classes))
        # set non-dropout layers to eval mode
        model.eval() 
        # set dropout layers to train mode
        enable_dropout(model)
        for batch, (x, y) in enumerate(loader):
            image = image.to(torch.device('cuda'))

            with torch.no_grad():
                output = model(x)
                prob = torch.sigmoid(output) # shape (n_samples, n_classes)
                prediction = (output > 0.5).float()
            preds = np.vstack((prob, output.cpu().numpy()))
        dropout_preds = np.vstack((dropout_preds,
                                         preds[np.newaxis, :, :]))
        # dropout preds - shape (forward_passes, n_samples, n_classes)
        
    # Calculating mean across forward passes
    mean = np.mean(dropout_preds, axis=0) # shape (n_samples, n_classes)

    # Calculating variance across forward passes
    variance = np.var(dropout_preds, axis=0) # shape (n_samples, n_classes)



def run_model(train=True):
    save_path = "/home/feliciaj/PolypSegmentation/saved_models/unet_dropout/"
    save_plot_path = "/home/feliciaj/PolypSegmentation/figures/"

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

    train_loader, val_loader, test_loader = data_loaders(
        batch_size=32, 
        train_transforms=train_transforms, 
        val_transforms=standard_transforms(256,256), 
        num_workers=4, 
        pin_memory=True
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 


    rates = [0, 0.3, 0.5]
    unets = []
    for rate in rates:
        unets.append(UNetClassifier(device=device, droprate=rate, max_epoch=10))

    if train:        
        # Training, set verbose=True to see loss after each epoch.
        [unet.train_model(train_loader, val_loader, verbose=True) for unet in unets]

        # Save trained models
        for rate, idx, unet in zip(rates, enumerate(unets)):
            torch.save(unet.model, save_path+"unet_"+str(idx)+".pt")

            dice = unet.val_dice
            # Prepare to save dice
            torch.save({
                f"unet_{rate}_dropout": dice.state_dict(),
            }, f"{save_plot_path}unet_dice_{idx}.pt")

            #unet.val_dice = list(map(str, unet.val_dice.numpy()))

        # Save test errors to plot figures 
        #open("unet_val_dices.txt","w").write("\n".join([",".join(unet.val_dice) for unet in unets])) 

    # Load saved models to CPU
    #unet_models = [torch.load(save_path+"unet_"+str(idx)+".pt", map_location={'cuda:0': 'cpu'}) for idx in [0,1]]

    # Load saved test errors to plot figures.
    #unet_val_dices = [array.split(",") for array in open("unet_val_dices.txt","r").read().split("\n")]
    #unet_val_dices = np.array(unet_val_dices, dtype="f")
    
    plt.figure(figsize=(8,7))
    for unet, idx, rate in zip(unets, enumerate(rates)):
        checkpoint = torch.load(f"{save_plot_path}unet_dice_{idx}.pt")
        dice = unet.load_state_dict(checkpoint[f"unet_{rate}_dropout"], map_location="cpu")
        label = f"U-Net with {rate:.2f}% dropout"
        plt.plot(range(1,len(dice)+1), dice, ".-", label=label)
        # force dropout layers to be on
        # loop thorugh test loader
        # predict
        # get dice scores
    plt.legend(loc="best");
    plt.xlabel("Epochs");
    plt.ylabel("Dice coefficient scores in validation set");
    plt.title("Dice coefficient scores on validation set from Kvasir-SEG Dataset for all Networks")
    plt.savefig("/home/feliciaj/PolypSegmentation/figures/unet.png")
    
    """

    labels = ["UNet no dropout","UNet with 30%% dropout","UNet with 50%% dropout"]

    plt.figure(figsize=(8, 7))
    for idx, d in enumerate(unet_test_dices.tolist()):
        plt.plot(range(1, len(d)+1), d, '.-', label=labels[idx], alpha=0.6);
    #plt.ylim([50, 250])
    plt.legend(loc="best");
    plt.xlabel("Epochs");
    plt.ylabel("Dice coefficient scores in validation set");
    plt.title("Dice coefficient scores on validation set from Kvasir-SEG Dataset for all Networks")
    plt.savefig("/home/feliciaj/PolypSegmentation/figures/unet.png")
    """

if __name__ == "__main__":
    run_model()