import torch
import torch.nn as nn
import torchvision
import os
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image

# local imports
from unet import UNet
from dataloader import data_loaders
from utils import save_grid, check_scores

from metrics import DiceLoss

#import matplotlib.pyplot as plt


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
    def __init__(self, modelA, modelB, modelC, modelD, modelE, device):
        super(MyEnsemble, self).__init__()

        self.modelA = modelA.to(device)
        self.modelB = modelB.to(device)
        self.modelC = modelC.to(device)
    
    def forward(self, x):
        shape = x.shape
        x1 = self.modelA(x.clone()) # pred from model A
        x2 = self.modelB(x.clone()) # pred from model B
        x3 = self.modelC(x.clone()) # pred from model C
        
        outputs = torch.stack([x1, x2, x3])
        
        mean = torch.mean(outputs, dim=0).double() # element wise mean from outout of ensemble models

        pred = torch.sigmoid(outputs)
        mean_pred = torch.sigmoid(mean).float() # only extract class prob

        variance = torch.mean((pred**2 - mean_pred), dim=0).double()

        return mean, variance 


def validate_ensembles():
    """
    Function loads trained models and make prediction on data from loader.
    Only supports for ensemble_size=5 and model="unet" for now.

    """
    save_folder = "/home/feliciaj/PolypSegmentation/ensembles/"
    load_folder = "/home/feliciaj/PolypSegmentation/saved_models/unet/"

    val_transforms = A.Compose(
        [   
            A.Normalize(
                mean=[0.5568, 0.3221, 0.2368],
                std=[0.3191, 0.2220, 0.1878],
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
    model = UNet(in_channels=3, out_channels=1)
    ensemble_size = 3
  
    model_list = []
    for i in range(ensemble_size):
        model_list.append(model)
    
    paths = os.listdir(load_folder)[:ensemble_size] # list of saved models in folder
    assert len(paths) == ensemble_size, "No. of folder elements does not match ensemble size"
    
    # load models
    for model, path in zip(model_list, paths):
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint["state_dict"])
    
    model.eval()

    model = MyEnsemble(
        model_list[0], model_list[1], model_list[2], device=device
    )

    criterion = DiceLoss()

    with torch.no_grad():
        for idx, (x, y) in enumerate(test_loader):
            mask = y.to(device=device).unsqueeze(1)
            x = x.to(device=device)
            prob, variance = model(x)
            pred = torch.sigmoid(prob)
            pred = (pred > 0.5).float() 
            variance = variance.cpu().detach()

            torchvision.utils.save_image(pred, f"{save_folder}/pred_{idx}.png")
            torchvision.utils.save_image(mask, f"{save_folder}/mask_{idx}.png")
            save_grid(variance, f"{save_folder}/heatmap_{idx}.png", rows=4, cols=8)

        loss, dice, iou = check_scores(test_loader, model, device, criterion)
    model.train()


if __name__ == "__main__":
    validate_ensembles()