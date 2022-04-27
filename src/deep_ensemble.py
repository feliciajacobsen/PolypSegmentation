import torch
import torch.nn as nn
import torchvision
import os
import sys
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

    def __init__(self, model, ensemble_size, device: str, load_folder: str):
        super().__init__()
        self.model = model
        self.ensemble_size = ensemble_size
        self.device = device
        self.load_folder = load_folder

    def forward(self, x):
        inputs = []
        for path in os.listdir(self.load_folder)[: self.ensemble_size]:
            checkpoint = torch.load(self.load_folder + path, map_location=self.device)
            self.model.load_state_dict(checkpoint["state_dict"])
            self.model.eval()
            for param in self.model.parameters():
                param.requires_grad_(False)
            inputs.append(self.model(x))

        outputs = torch.stack(inputs)  # (ensemble_size, b, c, w, h)
        sigmoided = torch.sigmoid(outputs)  # convert to probabilities
        mean = torch.mean(sigmoided, dim=0)  # take mean along stack dimension
        variance = torch.var(sigmoided, dim=0).double()  # torch.mean((sigmoided - mean) ** 2, dim=0).double()
        normalized_variance = (variance-torch.min(variance)) / (torch.max(variance)-torch.min(variance))

        return mean, normalized_variance


def test_ensembles(ensemble, device, loader, save_folder: str):
    """
    Function loads trained models and make prediction on data from loader.
    """

    dice, iou = 0, 0
    with torch.no_grad():
        for batch, (x, y) in enumerate(loader):
            y = y.to(device=device).unsqueeze(1)
            x = x.to(device=device)
            prob, variance = ensemble(x)
            pred = (prob > 0.5).float()
            dice += dice_coef(pred, y)
            iou += iou_score(pred, y)
            variance = variance.cpu().detach()

            # save images to save_folder
            torchvision.utils.save_image(
                pred, f"{save_folder}/pred_{batch}.png", nrow=5
            )
            torchvision.utils.save_image(y, f"{save_folder}/mask_{batch}.png", nrow=5)
            torchvision.utils.save_image(x, f"{save_folder}/input_{batch}.png", nrow=5)
            save_grid(
                variance.permute(0, 2, 3, 1),
                f"{save_folder}/heatmap_{batch}.png",
                rows=5,
                cols=5,
            )
    print(f"IoU score: {iou/len(loader)}")
    print(f"Dice score: {dice/len(loader)}")


def get_dice(ensemble_size, model, loader, device, load_folder):
    ensemble = DeepEnsemble(
        model=model, ensemble_size=ensemble_size, device=device, load_folder=load_folder
    )
    dice = 0
    with torch.no_grad():
        for batch, (x, y) in enumerate(loader):
            y = y.to(device).unsqueeze(1)
            x = x.to(device)
            prob, variance = ensemble(x)
            pred = (prob > 0.5).float()
            dice += dice_coef(pred, y)

    return dice / len(loader)

def dice_list():
    unet_BCE = [
        0.7659695108432987,
        0.7524669674465139,
        0.7827318302028493,
        0.7804074843632645,
        0.7900143614500293,
        0.7936320617880511,
        0.7955622496245283,
        0.8001379502345844,
        0.8000305039251812,
        0.8020472540287227,
        0.800647655987904,
        0.8035007418826028,
        0.8053836061785381,
        0.8047402817327324,
        0.8036835469731509,
        0.8034221510448492
    ]

    unet_dice = [
        0.8049752275876777, 
        0.806390636469403, 
        0.8132475716883494, 
        0.8140264594555717, 
        0.8152885230053997,
        0.8175577051451125,
        0.8182383594944287,
        0.8174845107364097,
        0.8178486270848578,
        0.8177954934707732,
        0.8178303571871943,
        0.8169476100163593,
        0.817242228176009,
        0.8174234227456949,
        0.8170245983639932,
        0.8171624245168824
    ]

    resunetplusplus_dice = [
        0.6953401563935416,
        0.7108269197580678,
        0.7215404303833568,
        0.7336649727418734,
        0.729759778933403,
        0.7307517664564741,
        0.7323681972426732,
        0.7375512516553291,
        0.7402523873031486,
        0.7364699345017984,
        0.7382709407606038,
        0.7425531671662954,
        0.7454024692650959,
        0.745000043328777,
        0.7431514971522293,
        0.7434531716286275
    ]

    resunetplusplus_BCE = [
        0.2996079171214447,
        0.3306244298015081,
        0.3228024694281624,
        0.34698293334304636,
        0.35120765247748564,
        0.3498512793232355,
        0.3454878134731113, 
        0.34813528426129253, 
        0.3332382677162492, 
        0.3342971379598024,
        0.3306325944364002,
        0.32356674419668313,
        0.3259832173221119,
        0.3298690111203878,
        0.33196417633060415,
        0.33459621990629695
    ]

    unet_dropout_dice = [
        0.7729696255613752, 
        0.7731290201825816, 
        0.7732157485887791, 
        0.7731296698771597, 
        0.7729378863174261, 
        0.7732144329922906, 
        0.7731877811587675, 
        0.7730678559871882, 
        0.7732069613942121, 
        0.7731418096160981, 
        0.773207496222386, 
        0.7731060664851418, 
        0.77311611262056, 
        0.7736303325912585, 
        0.7732633001381914, 
        0.7731157896008835
    ]

    unet_dropout_bce = [
        0.77594231632151, 
        0.7758483832497454, 
        0.7761015057019928, 
        0.7762778448032379, 
        0.7760717055307651, 
        0.7759592302793221, 
        0.7756697593855593,
         0.7756638080299914, 
        0.7754555193831894,
         0.7759654494746513, 
        0.7755255057126431, 
        0.7757711731685435, 
        0.7756915490764887, 
        0.7756499144294388, 
        0.7760246922279809, 
        0.7759334599290217
    ]

    resunetplusplus_dropout_dice = []
    
    resunetplusplus_dropout_BCE = []


    return unet_BCE, unet_dice, resunetplusplus_dice, resunetplusplus_BCE, unet_dropout_dice, unet_dropout_bce


def plot_ensembles_vs_score(save_plot_folder, filename, title):
    unet_bce, unet_dice, resunetplusplus_dice = dice_list()
    plt.figure(figsize=(8, 7))
    plt.plot(range(1,  len(unet_bce)+1), unet_bce, ".-", label="U-Net trained with BCE")
    plt.plot(range(1,  len(unet_dice)+1), unet_dice, "r.-", label="U-Net trained with DSC")
    plt.plot(range(1,  len(resunetplusplus_dice)+1), resunetplusplus_dice, "g.-", label="ResUnet++ trained with DSC")
    plt.legend(loc="best")
    plt.grid(ls="dashed", alpha=0.7)
    plt.xticks(range(1,  len(unet_bce)+1))
    plt.xlabel("Ensemble size")
    plt.ylabel("DSC")
    plt.title(title)
    plt.savefig(save_plot_folder + filename + ".png")




def run_ensembles(number):
    # print(f"This is a sanity check, random number is : {torch.rand(1)}")

    _, _, test_loader = data_loaders(
        batch_size=25,
        train_transforms=standard_transforms(256, 256),
        val_transforms=standard_transforms(256, 256),
        num_workers=4,
        pin_memory=True,
    )

    main_root = "/home/feliciaj/PolypSegmentation/"
    save_folder = main_root + "/results/results_kvasir/ensembles_resunet++_BCE/" 
    load_folder = main_root + "saved_models/resunet++_BCE/" 


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResUnetPlusPlus(in_channels=3, out_channels=1).to(device)  
    #model = UNet(in_channels=3, out_channels=1).to(device) 
    ensemble_size = number
    ensemble = DeepEnsemble(model, ensemble_size, device, load_folder)
    test_ensembles(ensemble, device, test_loader, save_folder)


    # print dice
    dice = get_dice(ensemble_size, model, test_loader, device, load_folder)
    print(f"Dice={dice}, ensemble size={number}")


    # make plot of all ensembles
    save_plot_folder = main_root + "/results/results_kvasir/plots/ensembles/"
    filename = "unet_ensembles_vs_score"
    title = "Different Deep Ensembles tested on Kvasir-SEG"

    #plot_ensembles_vs_score(save_plot_folder, filename, title)



if __name__ == "__main__":
    number = int(sys.argv[1])
    run_ensembles(number)
