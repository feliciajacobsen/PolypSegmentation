import torch
import torch.nn as nn
import torchvision
import os
import matplotlib.pyplot as plt
import numpy as np

# local imports
from unet import UNet
from resunetplusplus import ResUnetPlusPlus
from utils.dataloader import data_loaders, etis_larib_loader, cvc_clinic_loader
from utils.utils import save_grid, standard_transforms, get_class_weights
from utils.metrics import dice_coef, iou_score


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

    def __init__(self, model, ensemble_size: int, device: str, load_folder: str):
        super().__init__()
        self.device = device
        self.load_folder = load_folder
        self.ensemble_size = ensemble_size
        self.model = model

    def forward(self, x):
        inputs = []
        for path in os.listdir(self.load_folder)[:ensemble_size]:
            checkpoint = torch.load(self.load_folder + path, map_location=self.device)
            self.model.load_state_dict(checkpoint["state_dict"])
            self.model.eval()   
            for param in self.model.parameters():
                    param.requires_grad_(False)       
            inputs.append(self.model(x))

        outputs = torch.stack(inputs)  # shape = (ensemble_size, b, c, w, h)
        sigmoided = torch.sigmoid(outputs) # convert to probabilities
        mean = torch.mean(sigmoided, dim=0) # take mean along stack dimension
        variance = torch.var(sigmoided, dim=0).double() #torch.mean((sigmoided - mean) ** 2, dim=0).double()
        normalized_variance = (variance - torch.min(variance)) / (torch.max(variance) - torch.min(variance))

        return mean, variance


class ValidateTrainTestEnsemble:
    def __init__(self, model, ensemble_size, device, loader, load_folder):
        self.model = model
        self.ensemble_size = ensemble_size
        self.device = device
        self.loader = loader
        self.load_folder = load_folder
        self.ensemble = DeepEnsemble(self.model, self.ensemble_size, self.device, self.load_folder)

    def get_dice_iou(self, loader):
        pass

    def test_ensembles(self, save_folder):
        """
        Function loads trained models and make prediction on data from loader.
        """
 
        dice, iou = 0, 0
        with torch.no_grad():
            for batch, (x, y) in enumerate(self.loader):
                y = y.to(device=self.device).unsqueeze(1)
                x = x.to(device=self.device)
                prob, variance = self.ensemble(x)
                pred = (prob > 0.5).float()
                dice += dice_coef(pred, y)
                iou += iou_score(pred, y)
                variance = variance.cpu().detach()

                # save images to save_folder
                torchvision.utils.save_image(pred, f"{save_folder}/pred_{batch}.png", nrow=5)
                torchvision.utils.save_image(y, f"{save_folder}/mask_{batch}.png", nrow=5)
                torchvision.utils.save_image(x, f"{save_folder}/input_{batch}.png", nrow=5)
                save_grid(
                    variance.permute(0, 2, 3, 1),
                    f"{save_folder}/heatmap_{batch}.png",
                    rows=5,
                    cols=5,
                )
        print(f"IoU score: {iou/len(self.loader)}")
        print(f"Dice score: {dice/len(self.loader)}")


    def plot_dice_vs_ensemble_size(self, save_plot_folder, title):
        dice_list = []
        with torch.no_grad():
            for i in range(self.ensemble_size):
                print(f" Ensemble size: {i+1} ".center(50, "-"))
                running_dice, running_NLL = 0, 0
                ensemble_model = DeepEnsemble(self.model, i + 1, self.device, self.load_folder)
                for batch, (x, y) in enumerate(self.loader):
                    y = y.to(device=self.device).unsqueeze(1)
                    x = x.to(device=self.device)
                    prob, variance = ensemble_model(x)
                    pred = (prob > 0.5).float()
                    dice = dice_coef(pred, y)
                    print(dice, end=", ")
                    running_dice += dice

                average_dice = running_dice / len(self.loader)
                print("average dice:", average_dice)
                dice_list.append(average_dice)
                print(dice_list)

        plt.figure(figsize=(8, 7))
        plt.plot(range(1, self.ensemble_size + 1), dice_list, ".-")
        #plt.legend(loc="best")
        plt.xlabel("Number of networks in ensemble")
        plt.ylabel("Score")
        plt.title(title)
        plt.savefig(save_plot_folder + "ensembles_vs_score.png")


if __name__ == "__main__":
    print(f"This is a sanity check, random number is : {torch.rand(1)}")

    data = "kvasir"
    
    _, _, kvasir_loader = data_loaders(
        batch_size=25,
        train_transforms=standard_transforms(256, 256),
        val_transforms=standard_transforms(256, 256),
        num_workers=4,
        pin_memory=True,
    )

    etis_loader = etis_larib_loader(
            batch_size=25,
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
    model = UNet(in_channels=3, out_channels=1).to(device) #ResUnetPlusPlus(in_channels=3, out_channels=1)  
    ensemble_size = 16

    obj = ValidateTrainTestEnsemble(model, ensemble_size, device, test_loader, load_folder)

    #obj.test_ensembles(save_folder)

    obj.plot_dice_vs_ensemble_size(save_plot_folder, title)
    

    