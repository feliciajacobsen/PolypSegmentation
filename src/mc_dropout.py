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

from utils import data_loaders, save_preds_as_imgs, check_scores
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
        self.test_dice = []
        self.test_iou = []
        
    def fit(self, trainloader, valloader, verbose=True):
        device = self.device

        X_test, y_test = iter(valloader).next()
        X_test = X_test.to(device)

        tqdm_loader = tqdm(trainloader) # make progress bar
        scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.max_epoch):
            if self.scheduler is not None:
            self.scheduler.step(mean_val_loss)
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
            self.test_dice.append(mean_val_dice)
            self.test_iou.append(mean_val_iou)
            self.loss_.append(mean_val_loss)
            
            if verbose:
                print("---------")
                print("Epoch {} ==> loss: {}".format(epoch+1, self.loss_[-1]))

        return self.test_dice
    
    def predict(self, x): 
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





def run_model():
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

    val_transforms = A.Compose(
        [   
            A.Resize(height=256, width=256),
            A.Normalize(
                mean=[0.5579, 0.3214, 0.2350],
                std=[0.3185, 0.2218, 0.1875],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

    train_loader, val_loader, test_loader = data_loaders(
        batch_size=32, 
        train_transforms=val_transforms, 
        val_transforms=val_transforms, 
        num_workers=4, 
        pin_memory=True
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

    # Define networks
    unets = [UNetClassifier(device=device, droprate=0, max_epoch=150),
            UNetClassifier(device=device, droprate=0.3, max_epoch=150),
            UNetClassifier(device=device, droprate=0.5, max_epoch=150),
            ]
            
    # Training, set verbose=True to see loss after each epoch.
    [unet.fit(train_loader, val_loader,verbose=True) for unet in unets]

    # Save trained models
    for idx, unet in enumerate(unets):
        torch.save(unet.model, "unet_"+str(idx)+".pt")
        # Prepare to save dice
        unet.test_dice = list(map(str, unet.test_dice))

    # Save test errors to plot figures
    open("unet_test_dices.txt","w").write("\n".join([",".join(unet.test_dice) for unet in unets])) 

    save_model_path = "/home/PolypSegmentation/saved_models/unet_dropout/"

    # Load saved models to CPU
    unet_models = [torch.load(save_model_path+"unet_"+str(idx)+".pt", map_location={'cuda:0': 'cpu'}) for idx in [0,1]]

    # Load saved test errors to plot figures.
    unet_test_dices = [error_array.split(",") for error_array in open("unet_test_dices.txt","r").read().split("\n")]
    unet_test_dices = np.array(unet_test_dices,dtype="f")

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


if __name__ == "__main__":
    run_model()