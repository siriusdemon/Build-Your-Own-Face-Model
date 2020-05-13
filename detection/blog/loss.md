# 损失函数

本篇是 Build Your Own Face Detection Model 的第五节。

### 1 >> 开始之前

再次地，由于这是一个 CenterFace/Centernet 速成指南，我对原有的损失函数进行了简化。原版实现在[这里](https://github.com/xingyizhou/CenterNet/blob/master/src/lib/models/losses.py)。

具体而言，我用`MSE`取代了论文中的交叉熵损失，并改进了原代码中的`RegLoss`的实现。

### 2 >> 实现RegLoss

`RegLoss`指回归损失。最早出现在`Cornernet`的代码中，`Centernet`沿用了这个命名。

`RegLoss`用在对偏移量，宽高，人脸关键点的回归上，内部使用了`SmoothL1Loss`，也即是论文中提到的公式。

在`models`中创建一个`loss.py`，写入以下代码。

```py
import torch.nn as nn

class RegLoss(nn.Module):
    """Regression loss for CenterFace, especially 
    for offset, size and landmarks
    """

    def __init__(self):
        super().__init__()
        self.loss = nn.SmoothL1Loss(reduction='sum')

    def forward(self, pred, gt):
        mask = gt > 0
        pred = pred[mask]
        gt = gt[mask]
        loss = self.loss(pred, gt)
        loss = loss / (mask.float().sum() + 1e-4)
        return loss
```

无论是`offset`，`size`还是`landmarks`，它们的`groundtruth`都是很稀疏的，所以我们需要取出有值的那些点，只对那些点进行损失计算。由于这些真值点都是大于0的数，所以可以上面前三行代码取得。

### 3 >>

愿凡有所得，皆得自利利他。