import torch
import torch.nn as nn
import numpy as np
from torch.nn.modules.loss import _Loss


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
        dice = dice_coef(pred, truth)

        return bce_loss + (1 - dice)


class DiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target, eps=1e-8):
        pred = torch.sigmoid(input)

        intersection = (pred*target).flatten(1).sum(1)
        total = pred.flatten(1).sum(1) + target.flatten(1).sum(1)

        dice = (2.0 * intersection + eps) / (total + intersection + eps)

        return 1 - torch.mean(dice, dim=0)


def dice_coef(preds, labels, per_image=False):
    """
    Binary Dice for foreground class
    binary: 1 foreground, 0 background

    Args:
        pred (tensor): 1D tensor containing predicted (sigmoided + thresholded) pixel values.
        target (tensor): 1D tensor contining grund truth pixel values.
    Returns:
        1D tensor
    """
    smooth = 0.0001
    
    if not per_image:
        preds, labels = (preds,), (labels,)
      
    dices = []
    for pred, label in zip(preds, labels):
        intersection = ((label == 1) & (pred == 1)).sum() # true positive
        FP  = ((label==0) & (pred==1)).sum() # false positive
        FN = ((label==1) & (pred==0)).sum() # false negative 
        dice = (2*float(intersection) + smooth) / (float(FN) + float(FP) + 2*float(intersection) + smooth)
        dices.append(dice)
    dice = mean(dices) # mean accross images if per_image
    return dice


def iou_score(preds, labels, EMPTY=1., ignore=None, per_image=False):
    """
    Binary IoU for foreground class
    binary: 1 foreground, 0 background

    Args:
        pred (tensor): 1D tensor containing predicted (sigmoided + thresholded) pixel values.
        target (tensor): 1D tensor contining grund truth pixel values.
    Returns:
        1D tensor


    credit: https://github.com/Erlemar/Understanding-Clouds-from-Satellite-Images/blob/master/losses/lovasz_losses.py
    """
    if not per_image:
        preds, labels = (preds,), (labels,)
    ious = []
    for pred, label in zip(preds, labels):
        intersection = ((label == 1) & (pred == 1)).sum()
        union = ((label == 1) | ((pred == 1) & (label != ignore))).sum()
        if not union:
            iou = EMPTY
        else:
            iou = float(intersection) / float(union)
        ious.append(iou)
    iou = mean(ious)    # mean accross images if per_image
    return iou    


# helper function
def mean(l, ignore_nan=False, empty=0):
    """
    nanmean compatible with generators.
    """
    l = iter(l)
    if ignore_nan:
        l = ifilterfalse(isnan, l)
    try:
        n = 1
        acc = next(l)
    except StopIteration:
        if empty == 'raise':
            raise ValueError('Empty mean')
        return empty
    for n, v in enumerate(l, 2):
        acc += v
    if n == 1:
        return acc
    return acc / n    