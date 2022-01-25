import torch
import torch.nn as nn



class BCEDiceLoss(nn.Module):
    """
    Args:
        weight: An array of shape [num_classes,]
        input: A tensor of shape [N, num_classes, *]
        target: A tensor of shape same with input
    Return:
        Loss tensor
    """
    def __init__(self, weight=None):
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
    """
    Args:
        weight: An array of shape [num_classes,]
        input: A tensor of shape [N, num_classes, *]
        target: A tensor of shape same with input
    Return:
        Loss tensor
    """
    def __init__(self, weight=None):
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