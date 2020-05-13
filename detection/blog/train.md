# 训练

本篇是 Build Your Own Face Detection Model 的第六节。

这一节，我们建立一个训练流水线。

### 1 >> 开始之前

由于这部分代码与人脸识别系列的代码非常接近，所以我会比较粗略地带过。

### 2 >> 训练流水线

#### 2.0 >> 导入依赖

创建`train.py`，写入：
```py
import os
import os.path as osp

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import Config as cfg
from models.loss import RegLoss
from models.mnet import get_mobile_net
from datasets import WiderFace
```

#### 2.1 >> 加载数据

在`config.py`中配置训练的参数，包括检查点，迭代次数，学习率等

```py
class Config:
    # 省略...
    # checkpoints
    checkpoints = 'checkpoints'
    restore = False
    restore_model = 'final.pth'

    # training
    epoch = 50
    lr = 5e-4
    batch_size = 24
    pin_memory = True
    num_workers = 4
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
```

然后是在`train.py`中加载数据
```py
# Data Setup
dataset = WiderFace(cfg.dataroot, cfg.annfile, cfg.sigma, cfg.downscale, cfg.insize, cfg.train_transforms)
dataloader = DataLoader(dataset, batch_size=cfg.batch_size, 
    pin_memory=cfg.pin_memory, num_workers=cfg.num_workers)
device = cfg.device
```

#### 2.2 >> 创建网络

在`config.py`中继续添加
```py
# Network Setup
net = get_mobile_net(10, {'hm':1, 'wh':2, 'lm':10, 'off':2}, head_conv=24)
```

#### 2.3 >> 优化器及损失

```py
# Training Setup
optimizer = optim.Adam(net.parameters(), lr=cfg.lr)
heatmap_loss = nn.MSELoss()
wh_loss = RegLoss()
off_loss = RegLoss()
lm_loss = RegLoss()
```

#### 2.4 >> 检查点

```py
# Checkpoints Setup
checkpoints = cfg.checkpoints
os.makedirs(checkpoints, exist_ok=True)

if cfg.restore:
    weights_path = osp.join(checkpoints, cfg.restore_model)
    net.load_state_dict(torch.load(weights_path, map_location=device))
    print(f"load weights from checkpoints: {cfg.restore_model}")
```

#### 2.5 >> 开始训练
```py
# Start training
net.train()
net.to(device)

for e in range(cfg.epoch):
    for data, labels in tqdm(dataloader, desc=f"Epoch {e}/{cfg.epoch}",
                             ascii=True, total=len(dataloader)):
        data = data.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        out = net(data)

        heatmaps = torch.cat([o['hm'].squeeze() for o in out], dim=0)
        l_heatmap = heatmap_loss(heatmaps, labels[:, 0])

        offs = torch.cat([o['off'].squeeze() for o in out], dim=0)
        l_off = off_loss(offs, labels[:, [1,2]])

        whs = torch.cat([o['wh'].squeeze() for o in out], dim=0)
        l_wh = wh_loss(whs, labels[:, [3,4]])

        lms = torch.cat([o['lm'].squeeze() for o in out], dim=0)
        l_lm = lm_loss(lms, labels[:, 5:])

        loss = l_heatmap + l_off + l_wh * 0.1 + l_lm * 0.1
        loss.backward()
        optimizer.step()

    print(f"Epoch {e}/{cfg.epoch}, heat: {l_heatmap:.6f}, off: {l_off:.6f}, size: {l_wh:.6f}, landmark: {l_lm:.6f}")

    backbone_path = osp.join(checkpoints, f"{e}.pth")
    torch.save(net.state_dict(), backbone_path)
```

由于模型的输出是存放在字典里面的，所以我们需要选取出所有的`hm`输出，再进行损失计算。

```py
heatmaps = torch.cat([o['hm'].squeeze() for o in out], dim=0)
l_heatmap = heatmap_loss(heatmaps, labels[:, 0])
```

其他的同理。

以上即是训练的流水线。 

### 3 >>

愿凡有所得，皆能自利利他。