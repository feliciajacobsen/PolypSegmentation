import torch
import torch.nn as nn
import torchvision
import os
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
from matplotlib import gridspec

from utils.metrics import dice_coef, iou_score
from utils.dataloader import data_loaders


def get_mean_std(loader):
    """
    This function is borrowed from:
    https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/Pytorch/Basics/pytorch_std_mean.py
    
    Works for images with and without color channels.
    
    """

    channels_sum, channels_squared_sum, num_batches = 0, 0, 0
    
    for data, _ in loader:
        channels_sum += torch.mean(data, dim=[0,2,3]) # don't sum across channel dim
        channels_squared_sum += torch.mean(data**2, dim=[0,2,3])
        num_batches += 1

    mean = channels_sum/num_batches
    std = (channels_squared_sum/num_batches-mean**2)**0.5

    return mean, std


def get_class_weights(loader):

    num_zeros, num_ones = 0,0 
    for data, labels in loader:
        num_zeros += torch.sum(labels==0)
        num_ones += torch.sum(labels==1)

    return num_zeros/len(loader), num_ones/len(loader)
        
        



def check_scores(loader, model, device, criterion):
    """
    Validate for one epoch. Prints accuracy, Dice/F1 and IoU score.

    Args:
        loader (object): iterable-style dataset.
        model (class): provides with a forward method.
        device (cuda object): cpu or gpu.
        criterion (function): scoring function.

    Returns:
        Mean loss, dice and iou over loader data.
    """
    num_correct = 0
    num_pixels = 0
    dice = 0
    iou = 0
    loss = []

    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device).unsqueeze(1)
            prob = torch.sigmoid(model(x))
            pred = (prob > 0.5).float()
            num_correct += (pred == y).sum()
            num_pixels += torch.numel(pred)
            dice += dice_coef(pred, y)
            iou += iou_score(pred, y)
            loss.append(criterion(model(x), y))
    
    print(f"Accuracy: {num_correct}/{num_pixels} or {num_correct/num_pixels*100:.2f} percent")
    print(f"IoU score: {iou/len(loader)}")
    print(f"Dice score: {dice/len(loader)}")
    
    model.train()

    return sum(loss)/len(loader), dice/len(loader), iou/len(loader)


def save_checkpoint(epoch, state, folder):
    print("Epoch %d => Saving checkpoint" % epoch)
    torch.save(state, folder)


def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])    


def save_preds_as_imgs(loader, model, folder, device="cpu"):
    """
    Function saves the predicted masks an stores in separate folder.
    """
    if not os.path.exists(folder):
        os.makedirs(folder)

    model.eval()
    for batch, (x,y) in enumerate(loader):
        x = x.to(device=device)
        with torch.no_grad():
            preds = torch.sigmoid(model(x)) 
            preds = (preds > 0.5).float() 
        torchvision.utils.save_image(preds, f"{folder}/pred_{batch}.png")
        torchvision.utils.save_image(y.unsqueeze(1), f"{folder}/mask_{batch}.png")
    model.train()


class EarlyStopping():
    """
    Early stopping to stop the training when the loss does not improve after
    certain epochs.
    """
    def __init__(self, patience=50, min_delta=0):
        """
        :param patience: how many epochs to wait before stopping when loss is
               not improving
        :param min_delta: minimum difference between new loss and old loss for
               new loss to be considered as an improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
    def __call__(self, val_loss):
        if self.best_loss == None:
            self.best_loss = val_loss
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
        elif self.best_loss - val_loss < self.min_delta:
            self.counter += 1
            print(f"INFO: Early stopping counter {self.counter} of {self.patience}")
            if self.counter >= self.patience:
                print('INFO: Early stopping')
                self.early_stop = True


def save_grid(ims, folder, rows=None, cols=None, colorbar=False):
    if rows is None != cols is None:
        raise ValueError("Set either both rows and cols or neither.")

    if rows is None:
        rows = len(ims)
        cols = 1
   
    if ims.shape[0] < rows*cols:
        fig,axarr = plt.subplots(1, rows, figsize=(10,8))
    else:
        fig,axarr = plt.subplots(rows, cols, figsize=(10,8))

    #fig.subplots_adjust(wspace=0, hspace=0)

    for ax,im in zip(axarr.ravel(), ims):
        img = ax.imshow(im, cmap="turbo", vmin=im.min(), vmax=im.max())
        ax.set_axis_off()
        ax.set_aspect("equal") 
    
    if colorbar == True:
        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.85, 0.15, 0.025, 0.7])
        fig.colorbar(img, cax=cbar_ax, ticks=[0, 0.5, 1.0])
    
    fig.savefig(folder, transparent=False)


def standard_transforms(height, width):
    transforms = A.Compose(
        [   
            A.Resize(height=height, width=width),
            A.Normalize(
                mean=[0.5568, 0.3221, 0.2368],
                std=[0.3191, 0.2220, 0.1878],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )
    return transforms



if __name__ == "__main__":
    val_transforms = A.Compose([A.Resize(height=240, width=240), A.Normalize(mean=[0,0,0],std=[1.0,1.0,1.0],max_pixel_value=255.0),ToTensorV2()])
    # training dataset is 80 percent of total dataset
    train_loader, val_loader = data_loader(64,1,False,val_transforms)
    print(get_mean_std(train_loader)) # (tensor([0.5579, 0.3214, 0.2350]), tensor([0.3185, 0.2218, 0.1875]))
