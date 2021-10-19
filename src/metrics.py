import torch
import torch.nn as nn



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

class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super().__init__()
    
    def forward(self, input,target):
        pred = torch.sigmoid(input).view(-1)
        truth = target.view(-1)

        # Dice coefficient
        dice_coef = (2.0 * (pred * truth).double().sum() + 1) / (
            pred.double().sum() + truth.double().sum() + 1
        )

        return (1 - dice_coef)



def dice_coef(pred, target):
    """
    Dice coefficient used as metric.
    """
    intersection = (pred*target).double().sum()
    union = pred.double().sum() + target.double().sum()

    return (2.0 * intersection + 1) / (union + 1)


def iou_score(pred, target):
    """
    IoU (Intersect over Union) used as metric.
    """
    intersection = (pred*target).double().sum()
    union = target.double().sum() + pred.double().sum() 

    return (intersection + 1) / (union + 1)