import torch
import torch.nn as nn
import torchvision
import os
import albumentations as A
from albumentations.pytorch import ToTensorV2
from dataloader import data_loader




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



class BCEDiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super().__init__()

    def forward(self, input, target):
        pred = torch.sigmoid(input).view(-1)
        truth = target.view(-1)

        # BCE loss
        bce_loss = nn.BCEWithLogitsLoss()(pred, truth).double() 

        # Dice Loss
        dice_coef = (2.0 * (pred * truth).double().sum() + 1) / (
            pred.double().sum() + truth.double().sum() + 1
        )

        return bce_loss + (1 - dice_coef)


def dice_coef(pred, target):

    return (2.0 * (pred * target).double().sum() + 1) / (
            pred.double().sum() + target.double().sum() + 1
        )


def iou_score(pred, target):
    intersection = (pred*target).double().sum()
    union = target.double().sum() + pred.double().sum() 

    return (intersection + 1) / (union + 1)



def check_accuracy(loader, model, device):
    """
    Validate for one epoch. Prints accuracy and Dice/F1 score.

    Args:
        loader (object): iterable-style dataset.
        model (class): provides with a forward method.
        device (cuda object): cpu or gpu.

    Returns:
        None

    """
    num_correct = 0
    num_pixels = 0
    dice = 0
    iou = 0

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
    
    print(
        f"Got {num_correct}/{num_pixels} with acc {num_correct/num_pixels*100:.2f}"
    )
    print(f"IoU score: {iou/len(loader)}")
    print(f"Dice score: {dice/len(loader)}")
    
    model.train()



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



class Plotter():
    def __init__(self, modelParam, config):
    
        self.bestvalmeasure = None
    
        self.modelParam = modelParam
        self.config = config
        # plt.figure()
        self.fig, self.ax = plt.subplots(1,2)
        # if not self.modelParam['inNotebook']:
        # plt.show()
        self.fig.show()
        sleep(0.1)
        self.ax[0].set_ylabel("Loss")
        self.ax[0].set_xlabel("epoch [#]")

        # train_line = self.ax.plot([],[],color='blue', label='Train', marker='.', linestyle="")
        # val_line   = self.ax.plot([], [], color='red', label='Validation', marker='.', linestyle="")

        train_line = self.ax[0].plot([],[],color="blue", label="Train", marker=".", linestyle="")
        val_line   = self.ax[0].plot([], [], color="red", label="Validation", marker=".", linestyle="")
        self.ax[0].legend(handles=[train_line[0], val_line[0]])
        self.ax[0].set_axisbelow(True)
        self.ax[0].grid()
        self.ax[0].set_ylim([0, 5])
        self.ax[1].grid()
        self.ax[1].set_ylim([0, 0.39])
        sleep(0.1)
        return


    def update(self, current_epoch, loss, mode):
        if mode=='train':
            color = 'b'
        else:
            color = 'r'

        if self.modelParam['inNotebook']:
            self.ax[0].scatter(current_epoch, loss, c=color)
            self.ax[0].set_ylim(bottom=0, top=self.ax[0].get_ylim()[1])
            self.fig.canvas.draw()
            sleep(0.1)
        else:
            self.ax[0].scatter(current_epoch, loss, c=color)
            self.ax[0].set_ylim(bottom=0, top=self.ax[0].get_ylim()[1])
            self.fig.canvas.draw()
            sleep(0.1)
            self.save()
        return
        
    def update_withval(self, current_epoch, loss, valmeasure, mode):
        if mode=="train":
            color = "b"
        else:
            color = "r"

        if self.bestvalmeasure is None:
          self.bestvalmeasure = valmeasure
        elif self.bestvalmeasure < valmeasure :
          self.bestvalmeasure = valmeasure
        print("\n\n\ncurrent best val measure and current valmeasure ",self.bestvalmeasure, valmeasure)

        if self.modelParam["inNotebook"]:
            self.ax[0].scatter(current_epoch, loss, c=color)
            self.ax[0].set_ylim(bottom=0, top=self.ax[0].get_ylim()[1])
            
            self.ax[1].scatter(current_epoch, valmeasure, c='r')
            self.ax[1].set_ylim(bottom=0, top=self.ax[1].get_ylim()[1])
            
            self.fig.canvas.draw()
            sleep(0.1)
        else:
            self.ax[0].scatter(current_epoch, loss, c=color)
            self.ax[0].set_ylim(bottom=0, top=self.ax[0].get_ylim()[1])
            
            self.ax[1].scatter(current_epoch, valmeasure, c='r')
            self.ax[1].set_ylim(bottom=0, top=0.5)
            
            self.fig.canvas.draw()
            sleep(0.1)
            self.save()
        return

    def save(self):
        # path = self._getPath()
        pt = "./loss_images/"
        if not os.path.isdir(pt):
          os.makedirs(pt)
        path = pt+self.modelParam["modelName"][:-1]
        self.fig.savefig(path+".png")
        return

    def _getPath(self):
        keys = self.config.keys()
        path = "loss_images/"
        first=1
        for key in keys:
            if first!=1:
                path += "_"
            else:
                first=0
            element = self.config[key]
            if isinstance(element, str):
                path += element
            elif isinstance(element, int):
                path += key+str(element)
            elif isinstance(element, float):
                path += key+str(element)
            elif isinstance(element, list):
                path += ""
                for elm in element:
                    path += str(elm)
            elif isinstance(element, dict):
                path += ""
                for elKey, elVal in element.items():
                    path += str(elKey) + str(elVal).replace(".", "_")
            else:
                raise Exception("Unknown element in config")
        return path


if __name__ == "__main__":
    val_transforms = A.Compose([A.Resize(height=240, width=240), A.Normalize(mean=[0,0,0],std=[1.0,1.0,1.0],max_pixel_value=255.0),ToTensorV2()])
    # training dataset is 80 percent of total dataset
    train_loader, val_loader = data_loader(64,1,False,val_transforms)
    print(get_mean_std(train_loader)) # (tensor([0.5579, 0.3214, 0.2350]), tensor([0.3185, 0.2218, 0.1875]))
