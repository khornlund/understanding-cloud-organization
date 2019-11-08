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
from torch.autograd import Variable

from .scheduler import rolloff


class UpdatableLoss(nn.Module):
    def set_epoch(self, epoch):
        pass


class AnnealingLoss(UpdatableLoss):
    def __init__(self, start_weight, end_weight, start_anneal, anneal_epochs):
        super().__init__()
        self.start_weight = start_weight
        self.end_weight = end_weight
        self.start_anneal = start_anneal
        self.anneal_epochs = anneal_epochs
        self.curve = self.get_curve(start_weight, end_weight, anneal_epochs)

    def get_curve(self, start_weight, end_weight, anneal_epochs):
        curve = rolloff(
            anneal_epochs,
            loc_factor=0.5,
            scale_factor=0.1,
            magnitude=start_weight - end_weight,
            offset=end_weight,
        )
        return curve

    def get_weight_for_epoch(self, epoch):
        if epoch < self.start_anneal:
            return self.start_weight
        if epoch < self.start_anneal + self.anneal_epochs:
            return self.start_weight * self.curve[epoch - self.start_anneal]
        return self.end_weight


class DiceLoss(UpdatableLoss):
    def __init__(self, eps: float = 1e-7, threshold: float = None, soften: float = 0):
        super().__init__()
        self.loss_fn = partial(dice, eps=eps, threshold=threshold, soften=soften)

    def forward(self, logits, targets):
        dice = self.loss_fn(logits, targets)
        return 1 - dice


class BCEDiceLoss(UpdatableLoss):
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


class BCELovaszLoss(UpdatableLoss):
    def __init__(
        self,
        bce_weight: float = 0.5,
        lovasz_weight: float = 0.5,
        per_image=True,
        per_class=False,
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
            self.lovasz_loss = LovaszLoss(per_image=per_image, per_class=per_class)

    def forward(self, outputs, targets):
        if self.bce_weight == 0:
            return self.dice_weight * self.lovasz_loss(outputs, targets)
        if self.lovasz_weight == 0:
            return self.bce_weight * self.bce_loss(outputs, targets)

        bce = self.bce_loss(outputs, targets)
        lovasz = self.lovasz_loss(outputs, targets)
        loss = self.bce_weight * bce + self.lovasz_weight * lovasz
        return {"loss": loss, "bce": bce, "lovasz": lovasz}


class AnnealingBCELovaszLoss(AnnealingLoss):
    """
    Mixed loss than decays the weight of BCE as training progresses.
    """

    def __init__(
        self,
        bce_weight: float = 0.5,
        lovasz_weight: float = 0.5,
        per_image: bool = True,
        end_weight: float = 0.5,
        start_anneal: int = 0,
        anneal_epochs: int = 0,
    ):
        super().__init__(bce_weight, end_weight, start_anneal, anneal_epochs)
        self.bce_weight = bce_weight
        self.lovasz_weight = lovasz_weight
        if self.bce_weight != 0:
            self.bce_loss = nn.BCEWithLogitsLoss()
        if self.lovasz_weight != 0:
            self.lovasz_loss = LovaszLoss(per_image=per_image)

    def set_epoch(self, epoch):
        self.bce_weight = self.get_weight_for_epoch(epoch)
        self.lovasz_weight = 1 - self.bce_weight

    def forward(self, outputs, targets):
        if self.bce_weight == 0:
            return self.dice_weight * self.lovasz_loss(outputs, targets)
        if self.lovasz_weight == 0:
            return self.bce_weight * self.bce_loss(outputs, targets)

        bce = self.bce_loss(outputs, targets)
        lovasz = self.lovasz_loss(outputs, targets)
        loss = self.bce_weight * bce + self.lovasz_weight * lovasz
        return {"loss": loss, "bce": bce, "lovasz": lovasz}


class IoULoss(UpdatableLoss):
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
    def __init__(
        self, per_image: bool = True, per_class: bool = True, ignore: bool = None
    ):
        super().__init__()
        self.per_class = per_class
        self.loss = partial(lovasz_hinge, per_image=per_image, ignore=ignore)

    def forward(self, outputs, targets):
        if not self.per_class:
            return self.loss(outputs, targets)

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
