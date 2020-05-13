import torch.nn as nn


class RegLoss(nn.Module):
    """Regression loss for CenterFace, especially 
    for offset, size and landmarks
    """

    def __init__(self):
        super().__init__()
        self.loss = nn.SmoothL1Loss(reduction='sum')

    def forward(self, pred, gt):
        mask = gt.gt(0)
        pred = pred[mask]
        gt = gt[mask]
        loss = self.loss(pred, gt)
        loss = loss / (mask.float().sum() + 1e-4)
        return loss