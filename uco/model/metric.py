from sklearn.metrics import precision_score, recall_score


def dice_mean(output, targets, threshold=0.5):
    d0 = dice_0(output, targets, threshold)
    d1 = dice_1(output, targets, threshold)
    d2 = dice_2(output, targets, threshold)
    d3 = dice_3(output, targets, threshold)
    return (d0 + d1 + d2 + d3) / 4


def dice_0(output, targets, threshold=0.5):
    return dice_c(0, output, targets, threshold)


def dice_1(output, targets, threshold=0.5):
    return dice_c(1, output, targets, threshold)


def dice_2(output, targets, threshold=0.5):
    return dice_c(2, output, targets, threshold)


def dice_3(output, targets, threshold=0.5):
    return dice_c(3, output, targets, threshold)


def dice_c(c, output, targets, threshold=0.5):
    B, C, H, W = targets.size()
    total = 0.0
    for b in range(B):
        total += dice_single_channel(output[b, c, :, :], targets[b, c, :, :], threshold)
    return total / B


def dice_single_channel(probability, truth, threshold, eps=1e-9):
    p = (probability.view(-1) > threshold).float()
    t = (truth.view(-1) > 0.5).float()
    dice = (2.0 * (p * t).sum() + eps) / (p.sum() + t.sum() + eps)
    return dice


def accuracy_0(output, targets, threshold=0.5):
    return accuracy(output, targets, 0, threshold)


def accuracy_1(output, targets, threshold=0.5):
    return accuracy(output, targets, 1, threshold)


def accuracy_2(output, targets, threshold=0.5):
    return accuracy(output, targets, 2, threshold)


def accuracy_3(output, targets, threshold=0.5):
    return accuracy(output, targets, 3, threshold)


def accuracy(output, targets, class_, threshold=0.5):
    preds = (output[:, class_] > threshold).float()
    return (preds == targets[:, class_]).float().mean()


def precision_0(output, targets, threshold=0.5):
    return precision(output, targets, 0, threshold)


def precision_1(output, targets, threshold=0.5):
    return precision(output, targets, 1, threshold)


def precision_2(output, targets, threshold=0.5):
    return precision(output, targets, 2, threshold)


def precision_3(output, targets, threshold=0.5):
    return precision(output, targets, 3, threshold)


def precision(output, targets, class_, threshold):
    preds = (output[:, class_] > threshold).float().cpu().numpy()
    targets = targets[:, class_].cpu().numpy()
    return precision_score(targets, preds)


def recall_0(output, targets, threshold=0.5):
    return recall(output, targets, 0, threshold)


def recall_1(output, targets, threshold=0.5):
    return recall(output, targets, 1, threshold)


def recall_2(output, targets, threshold=0.5):
    return recall(output, targets, 2, threshold)


def recall_3(output, targets, threshold=0.5):
    return recall(output, targets, 3, threshold)


def recall(output, targets, class_, threshold):
    preds = (output[:, class_] > threshold).float().cpu().numpy()
    targets = targets[:, class_].cpu().numpy()
    return recall_score(targets, preds)


def precision_0_20(outputs, targets):
    return precision_0(outputs, targets, threshold=0.20)


def precision_0_30(outputs, targets):
    return precision_0(outputs, targets, threshold=0.30)


def precision_0_40(outputs, targets):
    return precision_0(outputs, targets, threshold=0.40)


def precision_0_50(outputs, targets):
    return precision_0(outputs, targets, threshold=0.50)


def precision_0_60(outputs, targets):
    return precision_0(outputs, targets, threshold=0.60)


def precision_0_70(outputs, targets):
    return precision_0(outputs, targets, threshold=0.70)


def precision_0_80(outputs, targets):
    return precision_0(outputs, targets, threshold=0.80)


def precision_1_20(outputs, targets):
    return precision_1(outputs, targets, threshold=0.20)


def precision_1_30(outputs, targets):
    return precision_1(outputs, targets, threshold=0.30)


def precision_1_40(outputs, targets):
    return precision_1(outputs, targets, threshold=0.40)


def precision_1_50(outputs, targets):
    return precision_1(outputs, targets, threshold=0.50)


def precision_1_60(outputs, targets):
    return precision_1(outputs, targets, threshold=0.60)


def precision_1_70(outputs, targets):
    return precision_1(outputs, targets, threshold=0.70)


def precision_1_80(outputs, targets):
    return precision_1(outputs, targets, threshold=0.80)


def precision_2_20(outputs, targets):
    return precision_2(outputs, targets, threshold=0.20)


def precision_2_30(outputs, targets):
    return precision_2(outputs, targets, threshold=0.30)


def precision_2_40(outputs, targets):
    return precision_2(outputs, targets, threshold=0.40)


def precision_2_50(outputs, targets):
    return precision_2(outputs, targets, threshold=0.50)


def precision_2_60(outputs, targets):
    return precision_2(outputs, targets, threshold=0.60)


def precision_2_70(outputs, targets):
    return precision_2(outputs, targets, threshold=0.70)


def precision_2_80(outputs, targets):
    return precision_2(outputs, targets, threshold=0.80)


def precision_3_20(outputs, targets):
    return precision_3(outputs, targets, threshold=0.20)


def precision_3_30(outputs, targets):
    return precision_3(outputs, targets, threshold=0.30)


def precision_3_40(outputs, targets):
    return precision_3(outputs, targets, threshold=0.40)


def precision_3_50(outputs, targets):
    return precision_3(outputs, targets, threshold=0.50)


def precision_3_60(outputs, targets):
    return precision_3(outputs, targets, threshold=0.60)


def precision_3_70(outputs, targets):
    return precision_3(outputs, targets, threshold=0.70)


def precision_3_80(outputs, targets):
    return precision_3(outputs, targets, threshold=0.80)


def recall_0_20(outputs, targets):
    return recall_0(outputs, targets, threshold=0.20)


def recall_0_30(outputs, targets):
    return recall_0(outputs, targets, threshold=0.30)


def recall_0_40(outputs, targets):
    return recall_0(outputs, targets, threshold=0.40)


def recall_0_50(outputs, targets):
    return recall_0(outputs, targets, threshold=0.50)


def recall_0_60(outputs, targets):
    return recall_0(outputs, targets, threshold=0.60)


def recall_0_70(outputs, targets):
    return recall_0(outputs, targets, threshold=0.70)


def recall_0_80(outputs, targets):
    return recall_0(outputs, targets, threshold=0.80)


def recall_1_20(outputs, targets):
    return recall_1(outputs, targets, threshold=0.20)


def recall_1_30(outputs, targets):
    return recall_1(outputs, targets, threshold=0.30)


def recall_1_40(outputs, targets):
    return recall_1(outputs, targets, threshold=0.40)


def recall_1_50(outputs, targets):
    return recall_1(outputs, targets, threshold=0.50)


def recall_1_60(outputs, targets):
    return recall_1(outputs, targets, threshold=0.60)


def recall_1_70(outputs, targets):
    return recall_1(outputs, targets, threshold=0.70)


def recall_1_80(outputs, targets):
    return recall_1(outputs, targets, threshold=0.80)


def recall_2_20(outputs, targets):
    return recall_2(outputs, targets, threshold=0.20)


def recall_2_30(outputs, targets):
    return recall_2(outputs, targets, threshold=0.30)


def recall_2_40(outputs, targets):
    return recall_2(outputs, targets, threshold=0.40)


def recall_2_50(outputs, targets):
    return recall_2(outputs, targets, threshold=0.50)


def recall_2_60(outputs, targets):
    return recall_2(outputs, targets, threshold=0.60)


def recall_2_70(outputs, targets):
    return recall_2(outputs, targets, threshold=0.70)


def recall_2_80(outputs, targets):
    return recall_2(outputs, targets, threshold=0.80)


def recall_3_20(outputs, targets):
    return recall_3(outputs, targets, threshold=0.20)


def recall_3_30(outputs, targets):
    return recall_3(outputs, targets, threshold=0.30)


def recall_3_40(outputs, targets):
    return recall_3(outputs, targets, threshold=0.40)


def recall_3_50(outputs, targets):
    return recall_3(outputs, targets, threshold=0.50)


def recall_3_60(outputs, targets):
    return recall_3(outputs, targets, threshold=0.60)


def recall_3_70(outputs, targets):
    return recall_3(outputs, targets, threshold=0.70)


def recall_3_80(outputs, targets):
    return recall_3(outputs, targets, threshold=0.80)
