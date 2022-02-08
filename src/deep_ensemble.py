import torch
import torch.nn as nn
import torchvision
import os
import albumentations as A
from albumentations.pytorch import ToTensorV2

# local imports
from unet import UNet
from doubleunet import DoubleUNet
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
            self.model_list.append(model)

        for model in self.model_list:
            model.to(device)

    def forward(self, x):
        inputs = []
        for model in self.model_list:
            inputs.append(model(x.clone()))
        outputs = torch.stack(inputs)

        mean = torch.mean(outputs, dim=0).double()  # element wise mean from outout of ensemble models
        pred = torch.sigmoid(outputs)
        mean_pred = torch.sigmoid(mean).double()  # only extract class prob
        variance = torch.mean((pred**2 - mean_pred), dim=0).double()

        return mean, variance


def test_ensembles():
    """
    Function loads trained models and make prediction on data from loader.
    Only supports for ensemble_size=3.

    """
    save_folder = "/home/feliciaj/PolypSegmentation/ensembles/"
    load_folder = "/home/feliciaj/PolypSegmentation/saved_models/unet/"

    train_loader, val_loader, test_loader = data_loaders(
        batch_size=32,
        train_transforms=standard_transforms(256, 256),
        val_transforms=standard_transforms(256, 256),
        num_workers=4,
        pin_memory=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = DiceLoss()
    model = UNet(in_channels=3, out_channels=1)
    ensemble_size = 3

    paths = os.listdir(load_folder)[:ensemble_size]  # list of saved models in folder
    assert (
        len(paths) == ensemble_size
    ), "No. of folder elements does not match ensemble size"

    # load models
    for path in paths:
        checkpoint = torch.load(load_folder + path)
        model.load_state_dict(checkpoint["state_dict"], strict=False)

    model.eval()
    model = DeepEnsemble(model, ensemble_size, device)

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

    model.train()


if __name__ == "__main__":
    test_ensembles()
