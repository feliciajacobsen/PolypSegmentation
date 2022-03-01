import torch
import torch.nn as nn
import torchvision
import os
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
import numpy as np

# local imports
from unet import UNet
from doubleunet import DoubleUNet
from resunetplusplus import ResUnetPlusPlus
from utils.dataloader import data_loaders
from utils.utils import save_grid, standard_transforms
from utils.metrics import DiceLoss, dice_coef, iou_score


class DeepEnsemble(nn.Module):
    """
    Ensemble of pretrained models.

    Args:
        model (torch object): deep learning object.
        ensemble_size (int): no. of deep learning objects.
        device (cuda object): device load models from.

    Returns:
        mean_pred (tensor): mean predicted mask by ensemble models of size (B,C,H,W).
        variance (tensor): normalized variance tensor of predicted mask of size (B,C,H,W).
    """

    def __init__(self, model, ensemble_size: int, device: str):
        super(DeepEnsemble, self).__init__()

        self.model_list = []
        for i in range(ensemble_size):
            self.model_list.append(model.to(device))
       
    def forward(self, x):
        inputs = []
        for model in self.model_list:
            inputs.append(model(x.clone()))
        outputs = torch.stack(inputs) # shape – (ensemble_size, b, c, w, h)

        mean = torch.mean(outputs, dim=0).double()  # element wise mean from output of ensemble models
        pred = torch.sigmoid(outputs)
        mean_pred = torch.sigmoid(mean).double()  # only extract class prob
        variance = torch.mean((pred - mean_pred)**2 , dim=0).double()
    
        normalized_variance = (variance - torch.min(variance)) / (torch.max(variance) - torch.min(variance))

        return mean, normalized_variance



class ValidateTrainTestEnsemble():
    def __init__(self, model, ensemble_size, device, loaders):
        self.model = model
        self.ensemble_size = ensemble_size
        self.device = device
        self.loaders = loaders
        self.ensemble = DeepEnsemble(self.model, self.ensemble_size, self.device)

    def get_dice_iou(self, loader):
        pass

    def test_ensembles(self, save_folder, load_folder):
        """
        Function loads trained models and make prediction on data from loader.

        """

        train_loader, val_loader, test_loader = self.loaders

        self.paths = os.listdir(load_folder)[:ensemble_size]  # list of saved models in folder

        # load models
        for path in self.paths:
            checkpoint = torch.load(load_folder + path)
            self.model.load_state_dict(checkpoint["state_dict"], strict=False)

        self.model.eval()
        model = self.ensemble
        dice, iou = 0, 0
        with torch.no_grad():
            for batch, (x, y) in enumerate(test_loader):
                y = y.to(device=device).unsqueeze(1)
                x = x.to(device=device)
                prob, variance = model(x)
                pred = torch.sigmoid(prob)
                pred = (pred > 0.5).float()
                dice += dice_coef(pred, y)
                iou += iou_score(pred, y)
                variance = variance.cpu().detach()

                torchvision.utils.save_image(pred, f"{save_folder}/pred_{batch}.png")
                torchvision.utils.save_image(y, f"{save_folder}/mask_{batch}.png")
                save_grid(
                    variance.permute(0, 2, 3, 1),
                    f"{save_folder}/heatmap_{batch}.png",
                    rows=4,
                    cols=8,
                )

        print(f"IoU score: {iou/len(test_loader)}")
        print(f"Dice score: {dice/len(test_loader)}")

        self.model.train()

    def plot_dice_vs_ensemble_size(self, save_plot_folder, load_folder):
        paths = os.listdir(load_folder)[:self.ensemble_size]
        for path in paths:
            checkpoint = torch.load(load_folder + path)
            self.model.load_state_dict(checkpoint["state_dict"], strict=False)

        _, _, test_loader = self.loaders
        
        dice_list = []
        for i in range(self.ensemble_size):
            running_dice = 0
            ensemble_model = DeepEnsemble(self.model, i+1, self.device)
            self.model.eval()
            for batch, (x,y) in enumerate(test_loader):
                with torch.no_grad():
                    y = y.to(device=self.device).unsqueeze(1)
                    x = x.to(device=self.device)
                    prob, variance = ensemble_model(x)
                    pred = torch.sigmoid(prob)
                    pred = (pred > 0.5).float()
                    running_dice += dice_coef(pred, y).cpu().numpy()

            dice_list.append(running_dice/len(test_loader))

        plt.figure(figsize=(8, 7))
        plt.plot(range(1, self.ensemble_size + 1), dice_list, ".-", label="Dice coeff")
        plt.legend(loc="best")
        plt.xlabel("Number of networks in ensemble")
        plt.ylabel("Dice")
        plt.title(f"UNet on Kvasir-SEG test set with Dice as loss")
        plt.savefig(save_plot_folder + ".png")


if __name__ == "__main__":
    model_name = "unet"
    loaders = data_loaders(
            batch_size=32,
            train_transforms=standard_transforms(256, 256),
            val_transforms=standard_transforms(256, 256),
            num_workers=4,
            pin_memory=True,
            )
    save_folder = "/home/feliciaj/PolypSegmentation/results/" + model_name + "/"
    load_folder = "/home/feliciaj/PolypSegmentation/saved_models/" + model_name + "/"      
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResUnetPlusPlus(in_channels=3, out_channels=1) # UNet(in_channels=3, out_channels=1)
    ensemble_size = 15

    obj = ValidateTrainTestEnsemble(model, ensemble_size, device, loaders)
    
    obj.test_ensembles(save_folder, load_folder)
    
    save_plot_folder = "/home/feliciaj/PolypSegmentation/results/figures/"
    obj.plot_dice_vs_ensemble_size(save_plot_folder, load_folder)
