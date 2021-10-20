import torch
import torch.nn as nn
import torchvision
import os
import albumentations as A
from albumentations.pytorch import ToTensorV2
from dataloader import data_loader

from metrics import dice_coef, iou_score


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


def check_scores(loader, model, device, criterion):
    """
    Validate for one epoch. Prints accuracy, Dice/F1 and IoU score.

    Args:
        loader (object): iterable-style dataset.
        model (class): provides with a forward method.
        device (cuda object): cpu or gpu.
        criterion (function): scoring function.

    Returns:
        Mean loss over training data.
    """
    num_correct = 0
    num_pixels = 0
    dice = 0
    iou = 0
    loss = []

    model.eval()
    with torch.no_grad():
        for batch, (x, y) in enumerate(loader):
            x = x.to(device)
            y = y.to(device).unsqueeze(1)
            pred = torch.sigmoid(model(x))
            pred = (pred > 0.5).float()
            num_correct += (pred == y).sum()
            num_pixels += torch.numel(pred)
            dice += dice_coef(pred, y)
            iou += iou_score(pred, y)
            loss.append(criterion(pred, y).item())
    
    print(
        f"Got {num_correct}/{num_pixels} with acc {num_correct/num_pixels*100:.2f}"
    )
    print(f"IoU score: {iou/len(loader)}")
    print(f"Dice score: {dice/len(loader)}")
    
    model.train()

    return sum(loss)/len(loader)



def save_checkpoint(state, filename="my_checkpoint.pt"):
    print("=> Saving checkpoint")
    torch.save(state, filename)


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

    for idx, (x,y) in enumerate(loader):
        x = x.to(device=device)
        with torch.no_grad():
            preds = torch.sigmoid(model(x)) 
            preds = (preds > 0.5).float() 
        torchvision.utils.save_image(preds, f"{folder}/pred_{idx}.png")
        torchvision.utils.save_image(y.unsqueeze(1), f"{folder}/mask_{idx}.png")

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




if __name__ == "__main__":
    val_transforms = A.Compose([A.Resize(height=240, width=240), A.Normalize(mean=[0,0,0],std=[1.0,1.0,1.0],max_pixel_value=255.0),ToTensorV2()])
    # training dataset is 80 percent of total dataset
    train_loader, val_loader = data_loader(64,1,False,val_transforms)
    print(get_mean_std(train_loader)) # (tensor([0.5579, 0.3214, 0.2350]), tensor([0.3185, 0.2218, 0.1875]))
