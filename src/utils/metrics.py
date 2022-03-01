import torch
import torch.nn as nn
import numpy as np
from torch.nn.modules.loss import _Loss
import torch.nn.functional as F


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



"""
class DiceLoss(nn.Module):
    def __init__(self, weight=None):
        super().__init__()
    
    def forward(self, input, target):
        assert input.shape[0] == target.shape[0], "Prediction and GT batch size do not match"
        pred = torch.sigmoid(input).view(-1) # view(-1) flattens tensor
        truth = target.view(-1)
        
        return (1 - dice_coef(pred, truth))

def dice_coef(pred, target):

    intersection = (pred*target).sum()
    total = pred.sum() + target.sum()

    return (2.0 * intersection + 1) / (total + intersection + 1)
"""


def dice_coef(preds, labels, EMPTY=1., ignore=None, per_image=True):
    """
    Binary Dice for foreground class
    binary: 1 foreground, 0 background

    Args:
        pred (tensor): 1D tensor containing predicted (sigmoided + thresholded) pixel values.
        target (tensor): 1D tensor contining grund truth pixel values.
    Returns:
        1D tensor
    """

    if not per_image:
        preds, labels = (preds,), (labels,)
    dices = []
    for pred, label in zip(preds, labels):
        intersection = ((label == 1) & (pred == 1)).sum() # true positive

        FP  = ((label==0) & (pred==1)).sum() # false positive
        FN = ((label==1) & (pred==0)).sum() # false negative 

        dice = 2 * float(intersection) / float(FN + FP + 2*intersection) 
        dices.append(dice)
    dice = mean(dices)    # mean accross images if per_image
    return dice



def soft_dice_score(
    output: torch.Tensor, target: torch.Tensor, smooth: float = 0.0, eps: float = 1e-7, dims=None
) -> torch.Tensor:
    """
    :param output:
    :param target:
    :param smooth:
    :param eps:
    :return:
    Shape:
        - Input: :math:`(N, NC, *)` where :math:`*` means any number
            of additional dimensions
        - Target: :math:`(N, NC, *)`, same shape as the input
        - Output: scalar.
    """
    assert output.size() == target.size()
    if dims is not None:
        intersection = torch.sum(output * target, dim=dims)
        cardinality = torch.sum(output + target, dim=dims)
    else:
        intersection = torch.sum(output * target)
        cardinality = torch.sum(output + target)
    dice_score = (2.0 * intersection + smooth) / (cardinality + smooth).clamp_min(eps)
    return dice_score



class DiceLoss(_Loss):
    """
    Implementation of Dice loss for binary image segmentation task.
    
    credit: https://github.com/BloodAxe/pytorch-toolbelt/blob/6b7565a1a890c55c610a8993bd221bb65acc5530/pytorch_toolbelt/losses/dice.py
    """

    def __init__(
        self,
        log_loss=False,
        from_logits=True,
        smooth: float = 0.0,
        ignore_index=None,
        eps=1e-7,
    ):
        """
        :param log_loss: If True, loss computed as `-log(jaccard)`; otherwise `1 - jaccard`
        :param from_logits: If True assumes input is raw logits
        :param smooth:
        :param ignore_index: Label that indicates ignored pixels (does not contribute to loss)
        :param eps: Small epsilon for numerical stability
        """
        super(DiceLoss, self).__init__()

        self.from_logits = from_logits
        self.smooth = smooth
        self.eps = eps
        self.ignore_index = ignore_index
        self.log_loss = log_loss

    def forward(self, y_pred, y_true):
        """
        :param y_pred: NxCxHxW
        :param y_true: NxHxW
        :return: scalar
        """
        assert y_true.size(0) == y_pred.size(0)

        if self.from_logits:
            # Apply activations to get [0..1] class probabilities
            # Using Log-Exp as this gives more numerically stable result and does not cause vanishing gradient on
            # extreme values 0 and 1
                y_pred = F.logsigmoid(y_pred).exp()

        bs = y_true.size(0)
        dims = (0, 2)

        
        y_true = y_true.view(bs, 1, -1)
        y_pred = y_pred.view(bs, 1, -1)

        if self.ignore_index is not None:
            mask = y_true != self.ignore_index
            y_pred = y_pred * mask
            y_true = y_true * mask


        scores = soft_dice_score(y_pred, y_true.type_as(y_pred), smooth=self.smooth, eps=self.eps, dims=dims)

        if self.log_loss:
            loss = -torch.log(scores.clamp_min(self.eps))
        else:
            loss = 1.0 - scores

        # Dice loss is undefined for non-empty classes
        # So we zero contribution of channel that does not have true pixels
        # NOTE: A better workaround would be to use loss term `mean(y_pred)`
        # for this case, however it will be a modified jaccard loss

        mask = y_true.sum(dims) > 0
        loss *= mask.to(loss.dtype)

        return loss.mean()


def iou_score(preds, labels, EMPTY=1., ignore=None, per_image=True):
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