import torch
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



def check_accuracy(loader, model, device):
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    model.eval()

    
    with torch.no_grad():
        for batch, (x, y) in enumerate(loader):
            x = x.to(device)
            y = y.to(device).unsqueeze(1)
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)
            dice_score += (2 * (preds * y).sum()) / (
                (preds + y).sum() + 1e-8
            )
    
    print(
        f"Got {num_correct}/{num_pixels} with acc {num_correct/num_pixels*100:.2f}"
    )
    print(f"Dice score: {dice_score/len(loader)}")
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
        torchvision.utils.save_image(y.unsqueeze(1), f"{folder}/{idx}.png")

    model.train()



if __name__ == "__main__":
    val_transforms = A.Compose([A.Resize(height=240, width=240), A.Normalize(mean=[0,0,0],std=[1.0,1.0,1.0],max_pixel_value=255.0),ToTensorV2()])
    # training dataset is 80 percent of total dataset
    train_loader, val_loader = data_loader(64,1,False,val_transforms)
    print(get_mean_std(train_loader)) # (tensor([0.5579, 0.3214, 0.2350]), tensor([0.3185, 0.2218, 0.1875]))
