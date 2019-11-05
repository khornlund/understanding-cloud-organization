"""
https://github.com/asanakoy/kaggle_carvana_segmentation/blob/master/asanakoy/losses.py
https://github.com/catalyst-team/catalyst/blob/master/catalyst/dl/utils/criterion/dice.py
"""
from __future__ import print_function, division
from functools import partial
from itertools import filterfalse as ifilterfalse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss
from torch.autograd import Variable


def bce_loss(output, target):
    return F.binary_cross_entropy_with_logits(output, target)


class DiceLoss(nn.Module):
    def __init__(self, eps: float = 1e-7, threshold: float = None, soften: float = 0):
        super().__init__()
        self.loss_fn = partial(dice, eps=eps, threshold=threshold, soften=soften)

    def forward(self, logits, targets):
        dice = self.loss_fn(logits, targets)
        return 1 - dice


class BCEDiceLoss(nn.Module):
    def __init__(
        self,
        eps: float = 1e-7,
        threshold: float = None,
        bce_weight: float = 0.5,
        dice_weight: float = 0.5,
        soften: float = 0,
    ):
        super().__init__()

        if bce_weight == 0 and dice_weight == 0:
            raise ValueError(
                "Both bce_wight and dice_weight cannot be "
                "equal to 0 at the same time."
            )

        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

        if self.bce_weight != 0:
            self.bce_loss = nn.BCEWithLogitsLoss()

        if self.dice_weight != 0:
            self.dice_loss = DiceLoss(eps=eps, threshold=threshold, soften=soften)

    def forward(self, outputs, targets):
        if self.bce_weight == 0:
            return self.dice_weight * self.dice_loss(outputs, targets)
        if self.dice_weight == 0:
            return self.bce_weight * self.bce_loss(outputs, targets)

        bce = self.bce_loss(outputs, targets)
        dice = self.dice_loss(outputs, targets)
        loss = self.bce_weight * bce + self.dice_weight * dice
        return {"loss": loss, "bce": bce, "dice": dice}


class BCELovaszLoss(nn.Module):
    def __init__(
        self, bce_weight: float = 0.5, lovasz_weight: float = 0.5, per_image=True
    ):
        super().__init__()

        if bce_weight == 0 and lovasz_weight == 0:
            raise ValueError(
                "Both bce_wight and lovasz_weight cannot be "
                "equal to 0 at the same time."
            )

        self.bce_weight = bce_weight
        self.lovasz_weight = lovasz_weight

        if self.bce_weight != 0:
            self.bce_loss = nn.BCEWithLogitsLoss()

        if self.lovasz_weight != 0:
            self.lovasz_loss = LovaszLoss(per_image=per_image)

    def forward(self, outputs, targets):
        if self.bce_weight == 0:
            return self.dice_weight * self.lovasz_loss(outputs, targets)
        if self.lovasz_weight == 0:
            return self.bce_weight * self.bce_loss(outputs, targets)

        bce = self.bce_loss(outputs, targets)
        lovasz = self.lovasz_loss(outputs, targets)
        loss = self.bce_weight * bce + self.lovasz_weight * lovasz
        return {"loss": loss, "bce": bce, "lovasz": lovasz}


class SlidingBCEDiceLoss(BCEDiceLoss):
    def __init__(
        self,
        dice_step: float = 0.01,
        eps: float = 1e-7,
        threshold: float = None,
        bce_weight: float = 0.5,
        dice_weight: float = 0.5,
    ):
        self.dice_step = dice_step
        super().__init(eps, threshold, bce_weight, dice_weight)

    def step(self):
        self.dice_weight += self.dice_step
        self.bce_weight = 1 - self.dice_weight


class IoULoss(nn.Module):
    """
    Intersection over union (Jaccard) loss
    Args:
        eps (float): epsilon to avoid zero division
        threshold (float): threshold for outputs binarization
        activation (str): An torch.nn activation applied to the outputs.
            Must be one of ['none', 'Sigmoid', 'Softmax2d']
    """

    def __init__(self, eps: float = 1e-7, threshold: float = None):
        super().__init__()
        self.metric_fn = partial(iou, eps=eps, threshold=threshold)

    def forward(self, outputs, targets):
        iou = self.metric_fn(outputs, targets)
        return 1 - iou


class BinaryFocalLoss(_Loss):
    def __init__(
        self,
        alpha=0.5,
        gamma=2,
        ignore_index=None,
        reduction="mean",
        reduced=False,
        threshold=0.5,
    ):
        """
        :param alpha:
        :param gamma:
        :param ignore_index:
        :param reduced:
        :param threshold:
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        if reduced:
            self.focal_loss = partial(
                focal_loss_with_logits,
                alpha=None,
                gamma=gamma,
                threshold=threshold,
                reduction=reduction,
            )
        else:
            self.focal_loss = partial(
                focal_loss_with_logits, alpha=alpha, gamma=gamma, reduction=reduction
            )

    def forward(self, label_input, label_target):
        """Compute focal loss for binary classification problem.
        """
        label_target = label_target.view(-1)
        label_input = label_input.view(-1)

        if self.ignore_index is not None:
            # Filter predictions with ignore label from loss computation
            not_ignored = label_target != self.ignore_index
            label_input = label_input[not_ignored]
            label_target = label_target[not_ignored]

        loss = self.focal_loss(label_input, label_target)
        return loss


class FocalBCEDiceLoss(BCEDiceLoss):
    def __init__(
        self,
        alpha=0.5,
        gamma=2,
        ignore_index=None,
        reduction="mean",
        reduced=False,
        eps: float = 1e-7,
        threshold: float = None,
        bce_weight: float = 0.5,
        dice_weight: float = 0.5,
    ):
        super().__init__(eps, threshold, bce_weight, dice_weight)
        self.bce_loss = BinaryFocalLoss(
            alpha, gamma, ignore_index, reduction, reduced, threshold
        )


# -- Lovasz Loss ----------------------------------------------------------------------

"""
Lovasz-Softmax and Jaccard hinge loss in PyTorch
Maxim Berman 2018 ESAT-PSI KU Leuven (MIT License)
https://github.com/bermanmaxim/LovaszSoftmax/blob/master/pytorch/lovasz_losses.py
"""


def lovasz_grad(gt_sorted):
    """
    Computes gradient of the Lovasz extension w.r.t sorted errors
    See Alg. 1 in paper
    """
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1.0 - intersection / union
    if p > 1:  # cover 1-pixel case
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


def iou_binary(preds, labels, EMPTY=1.0, ignore=None, per_image=True):
    """
    IoU for foreground class
    binary: 1 foreground, 0 background
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
    iou = mean(ious)  # mean accross images if per_image
    return 100 * iou


def iou(preds, labels, C, EMPTY=1.0, ignore=None, per_image=False):
    """
    Array of IoU for each (non ignored) class
    """
    if not per_image:
        preds, labels = (preds,), (labels,)
    ious = []
    for pred, label in zip(preds, labels):
        iou = []
        for i in range(C):
            # The ignored label is sometimes among predicted classes (ENet - CityScapes)
            if i != ignore:
                intersection = ((label == i) & (pred == i)).sum()
                union = ((label == i) | ((pred == i) & (label != ignore))).sum()
                if not union:
                    iou.append(EMPTY)
                else:
                    iou.append(float(intersection) / float(union))
        ious.append(iou)
    ious = [mean(iou) for iou in zip(*ious)]  # mean accross images if per_image
    return 100 * np.array(ious)


class LovaszLoss(nn.Module):
    def __init__(self, per_image: bool = True, ignore: bool = None):
        super().__init__()
        self.loss = partial(lovasz_hinge, per_image=per_image, ignore=ignore)

    def forward(self, outputs, targets):
        B, C, H, W = outputs.size()
        per_class_losses = torch.stack(
            [
                self.loss(logits=outputs[:, c, :, :], labels=targets[:, c, :, :])
                for c in range(C)
            ]
        )
        return per_class_losses.mean()


def lovasz_hinge(logits, labels, per_image=True, ignore=None):
    r"""
    Binary Lovasz hinge loss
      logits: [B, H, W] Variable, logits at each pixel (between -\infty and +\infty)
      labels: [B, H, W] Tensor, binary ground truth masks (0 or 1)
      per_image: compute the loss per image instead of per batch
      ignore: void class id
    """
    if per_image:
        loss = mean(
            lovasz_hinge_flat(
                *flatten_binary_scores(log.unsqueeze(0), lab.unsqueeze(0), ignore)
            )
            for log, lab in zip(logits, labels)
        )
    else:
        loss = lovasz_hinge_flat(*flatten_binary_scores(logits, labels, ignore))
    return loss


def lovasz_hinge_flat(logits, labels):
    r"""
    Binary Lovasz hinge loss
      logits: [P] Variable, logits at each prediction (between -\infty and +\infty)
      labels: [P] Tensor, binary ground truth labels (0 or 1)
      ignore: label to ignore
    """
    if len(labels) == 0:
        # only void pixels, the gradients should be 0
        return logits.sum() * 0.0
    signs = 2.0 * labels.float() - 1.0
    errors = 1.0 - logits * Variable(signs)
    errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
    perm = perm.data
    gt_sorted = labels[perm]
    grad = lovasz_grad(gt_sorted)
    loss = torch.dot(F.relu(errors_sorted), Variable(grad))
    return loss


def flatten_binary_scores(scores, labels, ignore=None):
    """
    Flattens predictions in the batch (binary case)
    Remove labels equal to 'ignore'
    """
    scores = scores.view(-1)
    labels = labels.view(-1)
    if ignore is None:
        return scores, labels
    valid = labels != ignore
    vscores = scores[valid]
    vlabels = labels[valid]
    return vscores, vlabels


class StableBCELoss(torch.nn.modules.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        neg_abs = -input.abs()
        loss = input.clamp(min=0) - input * target + (1 + neg_abs.exp()).log()
        return loss.mean()


def binary_xloss(logits, labels, ignore=None):
    r"""
    Binary Cross entropy loss
      logits: [B, H, W] Variable, logits at each pixel (between -\infty and +\infty)
      labels: [B, H, W] Tensor, binary ground truth masks (0 or 1)
      ignore: void class id
    """
    logits, labels = flatten_binary_scores(logits, labels, ignore)
    loss = StableBCELoss()(logits, Variable(labels.float()))
    return loss


def xloss(logits, labels, ignore=None):
    """
    Cross entropy loss
    """
    return F.cross_entropy(logits, Variable(labels), ignore_index=255)


# -- utils ----------------------------------------------------------------------------


class LabelSmoother:
    """
    Maps binary labels (0, 1) to (eps, 1 - eps)
    """

    def __init__(self, eps=1e-8):
        self.eps = eps
        self.scale = 1 - 2 * self.eps
        self.bias = self.eps / self.scale

    def __call__(self, t):
        return (t + self.bias) * self.scale


def dice(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    eps: float = 1e-7,
    threshold: float = None,
    soften: float = 0,
):
    outputs = torch.sigmoid(outputs)
    if threshold is not None:
        outputs = (outputs > threshold).float()
    intersection = torch.sum(targets * outputs)
    union = torch.sum(targets) + torch.sum(outputs)
    dice = (2 * intersection + soften) / (union + eps + soften)
    return dice


def focal_loss_with_logits(
    input: torch.Tensor,
    target: torch.Tensor,
    gamma=2.0,
    alpha: float = 0.25,
    reduction="mean",
    normalized=False,
    threshold: float = None,
) -> torch.Tensor:
    """
    https://github.com/BloodAxe/pytorch-toolbelt/blob/develop/pytorch_toolbelt/losses/functional.py
    Compute binary focal loss between target and output logits.
    See :class:`~pytorch_toolbelt.losses.FocalLoss` for details.
    References::
        https://github.com/open-mmlab/mmdetection/blob/master/mmdet/core/loss/losses.py
    """
    target = target.type(input.type())

    logpt = -F.binary_cross_entropy_with_logits(input, target, reduction="none")
    pt = torch.exp(logpt)

    # compute the loss
    if threshold is None:
        focal_term = (1 - pt).pow(gamma)
    else:
        focal_term = ((1.0 - pt) / threshold).pow(gamma)
        focal_term[pt < threshold] = 1

    loss = -focal_term * logpt

    if alpha is not None:
        loss = loss * (alpha * target + (1 - alpha) * (1 - target))

    if normalized:
        norm_factor = focal_term.sum()
        loss = loss / norm_factor

    if reduction == "mean":
        loss = loss.mean()
    if reduction == "sum":
        loss = loss.sum()
    if reduction == "batchwise_mean":
        loss = loss.sum(0)

    return loss


def isnan(x):
    return x != x


def mean(it, ignore_nan=False, empty=0):
    """
    nanmean compatible with generators.
    """
    it = iter(it)
    if ignore_nan:
        it = ifilterfalse(isnan, it)
    try:
        n = 1
        acc = next(it)
    except StopIteration:
        if empty == "raise":
            raise ValueError("Empty mean")
        return empty
    for n, v in enumerate(it, 2):
        acc += v
    if n == 1:
        return acc
    return acc / n
