""" Resnet_IR_SE in ArcFace """

import torch
import torch.nn as nn
import torch.nn.functional as F


class Flatten(nn.Module):
    def forward(self, x):
        return x.reshape(x.shape[0], -1)


class SEConv(nn.Module):
    """Use Convolution instead of FullyConnection in SE"""

    def __init__(self, channels, reduction):
        super().__init__()
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, kernel_size=1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x) * x


class SE(nn.Module):

    def __init__(self, channels, reduction):
        super().__init__()
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x) * x


class IRSE(nn.Module):

    def __init__(self, channels, depth, stride):
        super().__init__()
        if channels == depth:
            self.shortcut = nn.MaxPool2d(kernel_size=1, stride=stride)
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(channels, depth, (1, 1), stride, bias=False), 
                nn.BatchNorm2d(depth),
            )
        self.residual = nn.Sequential(
            nn.BatchNorm2d(channels),
            nn.Conv2d(channels, depth, (3, 3), 1, 1, bias=False),
            nn.PReLU(depth),
            nn.Conv2d(depth, depth, (3, 3), stride, 1, bias=False),
            nn.BatchNorm2d(depth),
            SEConv(depth, 16),
        )

    def forward(self, x):
        return self.shortcut(x) + self.residual(x)

class ResIRSE(nn.Module):
    """Resnet50-IRSE backbone"""

    def __init__(self, embedding_size, drop_ratio):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Conv2d(1, 64, (3, 3), 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.PReLU(64),
        )
        self.output_layer = nn.Sequential(
            nn.BatchNorm2d(512),
            nn.Dropout(drop_ratio),
            Flatten(),
            nn.Linear(512 * 8 * 8, embedding_size),
            nn.BatchNorm1d(embedding_size),
        )

        # ["channels", "depth", "stride"],
        self.res50_arch = [
            [64, 64, 2], [64, 64, 1], [64, 64, 1],
            [64, 128, 2], [128, 128, 1], [128, 128, 1], [128, 128, 1],
            [128, 256, 2], [256, 256, 1], [256, 256, 1], [256, 256, 1], [256, 256, 1],
            [256, 256, 1], [256, 256, 1], [256, 256, 1], [256, 256, 1], [256, 256, 1],
            [256, 256, 1], [256, 256, 1], [256, 256, 1], [256, 256, 1],
            [256, 512, 2], [512, 512, 1], [512, 512, 1],
        ]

        self.body = nn.Sequential(*[ IRSE(a,b,c) for (a,b,c) in self.res50_arch ])

    def forward(self, x):
        x = self.input_layer(x)
        x = self.body(x)
        x = self.output_layer(x) 
        return x


if __name__ == "__main__":
    from PIL import Image
    import numpy as np

    x = Image.open("../samples/009.jpg").convert('L')
    x = x.resize((128, 128))
    x = np.asarray(x, dtype=np.float32)    
    x = x[None, None, ...]
    x = torch.from_numpy(x)
    net = ResIRSE(512, 0.6)
    net.eval()
    with torch.no_grad():
        out = net(x)
    print(out.shape)