# -----------------------------------------------------------------------------------------------------------------------
# loss.py: This is the code for training loss used for PRN

# Usage: python src/deepglobe/refine-models/loss.py

# Note: To independently run, always run this file from PatchRefineNetwork (root) directory.
# -----------------------------------------------------------------------------------------------------------------------

from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F

class WeightedFocalLoss(nn.Module):
    "Non weighted version of Focal Loss"
    def __init__(self, alpha=.25, gamma=2):
        super(WeightedFocalLoss, self).__init__()
        self.alpha = torch.tensor([alpha, 1-alpha]).cuda()
        self.gamma = gamma

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        targets = targets.type(torch.long)
        at = self.alpha.gather(0, targets.data.view(-1))
        pt = torch.exp(-BCE_loss)
        F_loss = at*(1-pt)**self.gamma * BCE_loss
        return F_loss.mean()

class RegionSpecificLoss(nn.Module):
    def __init__(self, device, alpha_rst = 0.7, alpha_f = .25, gamma = 2):
        super(RegionSpecificLoss, self).__init__()
        self.alpha_f = alpha_f
        self.gamma = gamma
        self.alpha_rst = alpha_rst

        self.laplacian_kernel = torch.tensor([[-1., -1., -1.], [-1., 8., -1.], [-1., -1., -1.]], dtype=torch.float, requires_grad=False)
        self.laplacian_kernel = self.laplacian_kernel.view((1, 1, 3, 3))  # Shape format of weight for convolution
        self.laplacian_kernel = self.laplacian_kernel.to(device)

    def __call__(self, y_pred, y_gt):
        # Generate edge maps
        y_gt_edges = F.relu(torch.tanh(F.conv2d(y_gt, self.laplacian_kernel, padding=(1, 1))))
        y_pred_edges = F.relu(torch.tanh(F.conv2d(y_pred, self.laplacian_kernel, padding=(1, 1))))

        BCE_loss = F.binary_cross_entropy(y_pred, y_gt, reduction='none')
        targets = y_gt.type(torch.long)
        pt = torch.exp(-BCE_loss)
        F_loss = (1-self.alpha_f)*(1-pt)**self.gamma * BCE_loss
        focal_loss = F_loss.mean()

        edge_loss = F.binary_cross_entropy(input=y_pred_edges, target=y_gt_edges)

        total_loss = self.alpha_rst * focal_loss + (1 - self.alpha_rst) * edge_loss
        return total_loss

class RegularizationLoss(nn.Module):
    def __init__(self, device):
        super(RegularizationLoss, self).__init__()

    @staticmethod
    def weighted_bce(input_, target, weight_0=1.0, weight_1=1.0, eps=1e-15):
        wbce_loss = -weight_1 * target * torch.log(input_ + eps) - weight_0 * (1 - target) * torch.log(
            1 - input_ + eps)
        return torch.mean(wbce_loss)

    def __call__(self, y_pred, y_gt):
        reg_loss = self.weighted_bce(input_=y_pred, target=y_gt, weight_0=1.0, weight_1=1.12)
        return reg_loss

if __name__ == '__main__':
    if torch.cuda.is_available():
        device = torch.device(device='cuda')
    else:
        device = torch.device(device='cpu')

    dummy_input = torch.autograd.Variable(torch.sigmoid(torch.randn(1, 1, 512, 512)), requires_grad=True).to(device)
    dummy_gt = torch.autograd.Variable(torch.ones_like(dummy_input)).to(device)
    print('Input Size :', dummy_input.size())

    L_rs = RegionSpecificLoss(device=device)
    L_reg = RegularizationLoss(device=device)

    loss_rs = L_rs(dummy_input, dummy_gt)
    loss_reg =L_reg(dummy_input, dummy_gt)
    print('Loss_RS :', loss_rs, 'Loss_Reg :', loss_reg)

class bce_focal_loss(nn.Module):
    def __init__(self, batch=True):
        super(bce_focal_loss, self).__init__()
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
        return a