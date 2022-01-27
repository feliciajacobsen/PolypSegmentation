import torch
import torch.nn as nn
import torchvision
import os
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm

# local imports
from unet import UNet
from dataloader import data_loaders
from utils import (
    save_grid,  
    standard_transforms
)

from metrics import DiceLoss, dice_coef, iou_score


class MyEnsemble(nn.Module):
    """
    Ensemble of pretrained models.

    Args:
        model A-E (.pt-file): files of saved models.
        device (string): device to get models from.

    Returns:
        mean_pred (tensor): mean predicted mask by ensemble models of size (B,C,H,W).
        variance (tensor): normalized variance tensor of predicted mask of size (B,C,H,W).
    """
    def __init__(self, modelA, modelB, modelC, device):
        super(MyEnsemble, self).__init__()

        self.modelA = modelA.to(device)
        self.modelB = modelB.to(device)
        self.modelC = modelC.to(device)
    
    def forward(self, x):
        print("predict with model A")
        x1 = self.modelA(x.clone()) # pred from model A
        print("predict with model B")
        x2 = self.modelB(x.clone()) # pred from model B
        print("predict with model C")
        x3 = self.modelC(x.clone()) # pred from model C
        
        outputs = torch.stack([x1, x2, x3])
        mean = torch.mean(outputs, dim=0).double() # element wise mean from outout of ensemble models
        pred = torch.sigmoid(outputs)
        mean_pred = torch.sigmoid(mean).float() # only extract class prob
        variance = torch.mean((pred**2 - mean_pred), dim=0).double()

        return mean, variance 


def test_ensembles():
    """
    Function loads trained models and make prediction on data from loader.
    Only supports for ensemble_size=5 and model="unet" for now.

    """
    save_folder = "/home/feliciaj/PolypSegmentation/ensembles/"
    load_folder = "/home/feliciaj/PolypSegmentation/saved_models/unet/"

    train_loader, val_loader, test_loader = data_loaders(
        batch_size=32, 
        train_transforms=standard_transforms(256,256), 
        val_transforms=standard_transforms(256,256), 
        num_workers=4, 
        pin_memory=True
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = DiceLoss() 
    model = UNet(in_channels=3, out_channels=1)
    ensemble_size = 2
  
    model_list = []
    for i in range(ensemble_size):
        model_list.append(model)
    
    paths = os.listdir(load_folder)[:ensemble_size] # list of saved models in folder
    assert len(paths) == ensemble_size, "No. of folder elements does not match ensemble size"
    
    print("load models")
    # load models
    for model, path in zip(model_list, paths):
        checkpoint = torch.load(load_folder + path)
        model.load_state_dict(checkpoint["state_dict"])
    
    model.eval()

    model = MyEnsemble(
        model_list[0], model_list[1], model_list[2], device=device
    )
    tqdm_loader = tqdm(test_loader)
    dice = 0
    iou = 0
    print("test models")
    with torch.no_grad():
        for batch, (x, y) in enumerate(tqdm_loader):
            y = y.to(device=device).unsqueeze(1)
            x = x.to(device=device)
            print(x.shape)
            print(y.shape)
            prob, variance = model(x)
            print(variance.shape)
            pred = torch.sigmoid(prob)
            pred = (pred > 0.5).float() 
            dice += dice_coef(pred, y)
            iou += iou_score(pred, y)
            variance = variance.cpu().detach()

            torchvision.utils.save_image(pred, f"{save_folder}/pred_{batch}.png")
            torchvision.utils.save_image(y, f"{save_folder}/mask_{batch}.png")
            save_grid(variance.permute(0,2,3,1), f"{save_folder}/heatmap_{batch}.png", rows=4, cols=8)

    print(f"IoU score: {iou/len(test_loader)}")
    print(f"Dice score: {dice/len(test_loader)}")

    model.train()


if __name__ == "__main__":
    test_ensembles()