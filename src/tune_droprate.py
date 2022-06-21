import os
import torch
import torch.nn as nn
import torchvision
import torch.optim as optim
from torch.autograd import Variable
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
from tqdm import tqdm
import seaborn as sns

# import models
from unet import UNet_dropout
from resunetplusplus import ResUnetPlusPlus_dropout
from trainer import train_validate

# local imports
from utils.dataloader import data_loaders
from utils.utils import (
    check_scores, 
    save_grid, 
    standard_transforms
)
from utils.metrics import (
    dice_coef, 
    iou_score, 
    DiceLoss
)


class DropoutClassifier:
    def __init__(self, loaders, device, droprate=0.1, max_epoch=150, lr=1e-4):
        super(DropoutClassifier, self).__init__()
        self.loaders = loaders
        self.max_epoch = max_epoch
        self.droprate = droprate
        self.lr = lr
        self.model = ResUnetPlusPlus_dropout(in_channels=3, out_channels=1, droprate=droprate).to(device) #UNet_dropout(3, 1, droprate).to(device)
        self.device = device
        self.criterion = DiceLoss().to(device) #nn.BCEWithLogitsLoss().to(device) 
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.loss_ = []
        self.test_error = []
        self.test_accuracy = []
        
    def fit(self):
        train_loader, val_loader, _ = self.loaders
        scaler = torch.cuda.amp.GradScaler()
        dices = []
        for epoch in range(self.max_epoch):
            running_loss = []
            for i, (inputs, labels) in enumerate(train_loader):
                labels = labels.unsqueeze(1).to(self.device)
                inputs = inputs.to(self.device)
                
                with torch.cuda.amp.autocast():
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels)

                self.optimizer.zero_grad()  
                scaler.scale(loss).backward() 
                scaler.step(self.optimizer)  
                scaler.update() 
                running_loss.append(loss.item())
            self.loss_ = sum(running_loss) / len(train_loader)

            dices.append(self.get_dice(self.model))

            if epoch == self.max_epoch - 1:
                torch.save(dices, f"resunet_dsc_{self.droprate}.pt")

            print(f"Epoch {epoch+1}, loss: {self.loss_}")
    
    def get_dice(self, model):
        _, val_loader, _ = self.loaders
        model.eval()
        dice = 0
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(self.device)
                y = y.to(self.device).unsqueeze(1)
                prob = torch.sigmoid(model(x))
                pred = (prob > 0.5).float()
                dice += dice_coef(pred, y)
        
        model.train()

        return dice/len(val_loader)


def main():
    train_transforms = A.Compose(
        [
            A.Resize(height=256, width=256),
            A.Rotate(limit=35, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.Normalize(
                mean=[0.5579, 0.3214, 0.2350],
                std=[0.3185, 0.2218, 0.1875],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )
    val_transforms = standard_transforms(256, 256)
    loaders = data_loaders(
        batch_size=32,
        train_transforms=train_transforms,
        val_transforms=val_transforms,
        num_workers=4,
        pin_memory=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    droprates = [0, 0.1, 0.3, 0.5]
    
    # Save models
    for rate in droprates:
        model = DropoutClassifier(loaders=loaders, device=device, droprate=rate)
        model.fit()
    
    # Load saved models to CPU
    sns.set()
    plt.figure(figsize=(10, 7))
    for rate in droprates:
        dices = torch.load(f"/home/feliciaj/PolypSegmentation/dropout_rates/resunet_dsc_{rate}.pt",
                    map_location=torch.device("cpu"),
                )
        if rate == 0:
            label = "ResUNet++ no dropout"
        else:
            label = f"ResUNet++ dropout rate={rate:.1f}"
        plt.plot(range(1, 150+1), dices, ".-", label=label)
  
    plt.legend(loc="best")
    plt.xlabel('Epochs')
    plt.ylabel('DSC')
    plt.title('Dropout ResUNet++ trained with DSC loss on validation Kvasir-SEG')
    plt.savefig("/home/feliciaj/PolypSegmentation/results/plots/MC_dropout/resunet++_dsc_droprates.png")



if __name__ == "__main__":
    main()
    
    