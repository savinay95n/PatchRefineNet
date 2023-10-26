# -----------------------------------------------------------------------------------------------------------------------
# metrics.py: This is the code containing metrics to train base networks on kvasir
# -----------------------------------------------------------------------------------------------------------------------

import torch
import torch.nn as nn
from torch.autograd import Variable as V
import numpy as np

class dice_bce_loss(nn.Module):
    def __init__(self, batch=True):
        super(dice_bce_loss, self).__init__()
        self.batch = batch
        self.bce_loss = nn.BCELoss()
        
    def soft_dice_coeff(self, y_true, y_pred):
        smooth = 0.0  # may change
        if self.batch:
            i = torch.sum(y_true)
            j = torch.sum(y_pred)
            intersection = torch.sum(y_true * y_pred)
        else:
            i = y_true.sum(1).sum(1).sum(1)
            j = y_pred.sum(1).sum(1).sum(1)
            intersection = (y_true * y_pred).sum(1).sum(1).sum(1)
        score = (2. * intersection + smooth) / (i + j + smooth)
        return score.mean()

    def soft_dice_loss(self, y_true, y_pred):
        loss = 1 - self.soft_dice_coeff(y_true, y_pred)
        return loss

    def focal_loss(self, y_true, y_pred, alpha=0.8, gamma=2, smooth=1):
        BCE = self.bce_loss(y_true, y_pred)
        BCE_EXP = torch.exp(-BCE)
        loss = alpha * (1 - BCE_EXP) ** gamma * BCE
        return loss
        
    def __call__(self, y_true, y_pred):
        a =  self.bce_loss(y_pred, y_true)
        b =  self.soft_dice_loss(y_true, y_pred)
        c = self.focal_loss(y_true, y_pred)
        return a + b

def int_uni(gt, pred, DEVICE, smooth=1):
    pred_th = torch.zeros(pred.shape).to(DEVICE)
    pred_th[pred >= 0.5] = 1
    pred_th[pred < 0.5] = 0
    intersection = torch.sum(gt * pred_th)
    union = torch.sum(gt + pred_th) - intersection
    iou = torch.mean((intersection + smooth) / (union + smooth))
    return iou

def int_uni_boundary(gt, pred, DEVICE, smooth=1):
    pred_th = torch.zeros(pred.shape).to(DEVICE)
    pred_th[pred >= 0.01] = 1
    pred_th[pred < 0.01] = 0
    intersection = torch.sum(gt * pred_th)
    union = torch.sum(gt + pred_th) - intersection
    iou = torch.mean((intersection + smooth) / (union + smooth))
    return iou

def int_uni_numpy(gt, pred, smooth=1):
    intersection = np.sum(gt * pred)
    union = np.sum(gt + pred) - intersection
    iou = np.mean((intersection + smooth) / (union + smooth))
    return iou