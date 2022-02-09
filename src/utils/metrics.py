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

    Returns:
        1D tensor
    """
    def __init__(self, weight=None):
        super().__init__()
    
    def forward(self, input, target):
        assert input.shape[0] == target.shape[0], "Prediction and GT batch size do not match"
        pred = torch.sigmoid(input).view(-1) # view(-1) flattens tensor
        truth = target.view(-1)
        
        return (1 - dice_coef(pred, truth))

def dice_coef(pred, target):
    """
    Dice coefficient used as metric.

    Args:
        pred (tensor): 1D tensor containing predicted (sigmoided + thresholded) pixel values.
        target (tensor): 1D tensor contining grund truth pixel values.
    Returns:
        1D tensor
    """
    intersection = (pred*target).sum()
    union = pred.sum() + target.sum()

    return (2.0 * intersection) / (union+intersection)



def iou_score(pred, target):
    """
    IoU (Intersect over Union) used as metric.

    Args:
        pred (tensor): 1D tensor containing predicted (sigmoided + thresholded) pixel values.
        target (tensor): 1D tensor contining grund truth pixel values.
    Returns:
        1D tensor
    """
    intersection = (pred*target).double().sum()
    union = target.double().sum() + pred.double().sum() 

    return (intersection + 1) / (union + 1)