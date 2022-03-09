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
from utils.dataloader import data_loaders, etis_larib_loader, cvc_clinic_loader
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
        outputs = torch.stack(inputs)  # shape – (ensemble_size, b, c, w, h)

        mean = torch.mean(
            outputs, dim=0
        ).double()  # element wise mean from output of ensemble models
        pred = torch.sigmoid(outputs)
        mean_pred = torch.sigmoid(mean).double()  # only extract class prob
        variance = torch.mean((pred - mean_pred) ** 2, dim=0).double()

        normalized_variance = (variance - torch.min(variance)) / (torch.max(variance) - torch.min(variance))

        return mean, normalized_variance


class ValidateTrainTestEnsemble:
    def __init__(self, model, ensemble_size, device, loader):
        self.model = model
        self.ensemble_size = ensemble_size
        self.device = device
        self.loader = loader
        self.ensemble = DeepEnsemble(self.model, self.ensemble_size, self.device)

    def get_dice_iou(self, loader):
        pass

    def test_ensembles(self, save_folder, load_folder):
        """
        Function loads trained models and make prediction on data from loader.

        """

        # list of saved models in folder
        self.paths = os.listdir(load_folder)[:ensemble_size]  

        # load models
        for path in self.paths:
            checkpoint = torch.load(load_folder + path)
            self.model.load_state_dict(checkpoint["state_dict"], strict=False)

        self.model.eval()
        model = self.ensemble
        dice, iou = 0, 0
        with torch.no_grad():
            for batch, (x, y) in enumerate(self.loader):
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

        print(f"IoU score: {iou/len(self.loader)}")
        print(f"Dice score: {dice/len(self.loader)}")

        self.model.train()

    def plot_dice_vs_ensemble_size(self, save_plot_folder, load_folder, title):
        paths = os.listdir(load_folder)[: self.ensemble_size]
        for path in paths:
            checkpoint = torch.load(load_folder + path)
            self.model.load_state_dict(checkpoint["state_dict"], strict=False)

        dice_list = []
        for i in range(self.ensemble_size):
            running_dice = 0
            ensemble_model = DeepEnsemble(self.model, i + 1, self.device)
            self.model.eval()
            for batch, (x, y) in enumerate(self.loader):
                with torch.no_grad():
                    y = y.to(device=self.device).unsqueeze(1)
                    x = x.to(device=self.device)
                    prob, variance = ensemble_model(x)
                    pred = torch.sigmoid(prob)
                    pred = (pred > 0.5).float()
                    running_dice += dice_coef(pred, y).item()

            dice_list.append(running_dice / len(self.loader))

        print(dice_list)

        plt.figure(figsize=(8, 7))
        plt.plot(range(1, self.ensemble_size + 1), dice_list, ".-", label="Dice coeff")
        plt.legend(loc="best")
        plt.xlabel("Number of networks in ensemble")
        plt.ylabel("Dice")
        plt.title(title)
        plt.savefig(save_plot_folder + "hello2.png")


if __name__ == "__main__":

    data = "kvasir"
    
    _, _, kvasir_loader = data_loaders(
        batch_size=32,
        train_transforms=standard_transforms(256, 256),
        val_transforms=standard_transforms(256, 256),
        num_workers=4,
        pin_memory=True,
    )

    
    etis_loader = etis_larib_loader(
            batch_size=32,
            transforms=standard_transforms(256, 256),
            num_workers=4,
            pin_memory=True,
    )

    main_root = "/home/feliciaj/PolypSegmentation"
    load_folder = main_root + "/saved_models/unet_BCE/"

    if data=="etis":
        save_folder = main_root + "/results/results_etis/ensembles_unetBCE/"
        save_plot_folder = main_root + "/results/results_etis/plots/"
        title = "Deep Ensemble of UNets trained with BCE loss and tested on ETIS-Larib"
        test_loader = etis_loader

    elif data=="kvasir":
        save_folder = main_root + "/results/results_kvasir/ensembles_unetBCE/"
        save_plot_folder = main_root + "/results/results_kvasir/plots/"
        title = "Deep Ensemble of UNets trained with BCE loss and tested on Kvasir-SEG"
        test_loader = kvasir_loader

    else:
        save_folder = main_root + "/results/results_cvc/ensembles_unetBCE/"
        save_plot_folder = main_root + "/results/results_cvc/plots/"
        title = "Deep Ensemble of UNets trained with BCE loss and tested on CVC-ClinicDB"  
        #test_loader = cvc_loader

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(in_channels=3, out_channels=1) #ResUnetPlusPlus(in_channels=3, out_channels=1)  
    ensemble_size = 12

    obj = ValidateTrainTestEnsemble(model, ensemble_size, device, test_loader)

    obj.test_ensembles(save_folder, load_folder)

    obj.plot_dice_vs_ensemble_size(save_plot_folder, load_folder, title)
