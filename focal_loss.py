import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')

        pt = torch.exp(-ce_loss)

        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.alpha is not None:
            if isinstance(self.alpha, (float, int)):
                 focal_loss = self.alpha * focal_loss
            else:
                alpha = self.alpha
                if hasattr(alpha, 'device') and alpha.device != inputs.device:
                    alpha = alpha.to(inputs.device)

                if isinstance(alpha, torch.Tensor) and alpha.dim() == 1:
                    alpha_t = alpha[targets]
                    focal_loss = alpha_t * focal_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
