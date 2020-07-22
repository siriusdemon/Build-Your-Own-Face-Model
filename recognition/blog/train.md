# 训练

欢迎阅读！本文是 Build Your Own Face Recognition Model 系列博客的第五篇。

在这一节，我们将建立一个训练的流水线，将前面四节的内容整合起来，这一节的内容比较多，请给自己一些耐心！

### 1 >> 开始之前

目前我们已经拥有了

+ 数据集
+ 网络结构
+ 损失函数
+ 度量函数

训练一个人脸识别网络的要素都准备齐全了。训练的流水线如下：

```py
读取数据 -> 数据预处理 -> 模型得到数据，输出embedding -> embedding和标签进入度量函数，输出概率值 
-> 损失函数计算损失 -> 反向传播调整参数 -> 直到模型收敛
```

由于这样的流水线，每一处都有好多参数可以选择，我们将建立一个额外的配置文件，专门管理这些参数。

### 2 >> 配置文件

我们首先关心数据问题, 我将使用`torchvision`提供的一系列函数来做数据预处理。创建一个`config.py`，写入以下代码

```py
import torch
import torchvision.transforms as T

class Config:
    # dataset
    train_root = '/data/CASIA-WebFace'
    test_root = "/data/lfw-align-128"
    test_list = "/data/lfw_test_pair.txt"
```

我指定了自己的数据路径，你得根据你自己的路径配置。

### 3 >> 数据预处理

按理说，数据预处理应该单独一篇来说，不过因为我们的数据处理比较简单，我觉得合并到训练过程，整体感也比较好。

继续往`config.py`添加代码

```py
class Config: 
    # ... 前面省略 ...
    input_shape = [1, 128, 128]
    train_transform = T.Compose([
        T.Grayscale(),
        T.RandomHorizontalFlip(),
        T.Resize((144, 144)),
        T.RandomCrop(input_shape[1:]),
        T.ToTensor(),
        T.Normalize(mean=[0.5], std=[0.5]),
    ])
    test_transform = T.Compose([
        T.Grayscale(),
        T.Resize(input_shape[1:]),
        T.ToTensor(),
        T.Normalize(mean=[0.5], std=[0.5]),
    ])
```
我设计的输入是`1 x 128 x 128`，即输入灰度图，其大小为`128 x 128`。也有人的输入是`3 x 112 x 112`。

我的数据预处理函数分为训练和测试，在训练时，我的数据预处理流水线是这样子的

```py
灰度化 -> 随机水平翻转 -> 放缩为 144 x 144 -> 随机剪成 128 x 128 
-> 转为Pytorch训练的格式 -> 数据归一化
```

测试的时候，我不希望进行水平翻转，也不想做随机剪切，所以流水线是这样子的

```py
灰度化 -> 放缩成 128 x 128 -> 转为Pytorch训练的格式 -> 数据归一化
```

现在预处理的操作有了，我们将数据读取与预处理连起来！

创建一个`dataset.py`文件，写入以下代码

```py
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
```

这里我们导入了两个类，`ImageFolder`可以很方便地处理以文件夹进行分类的数据，如

```py
动物
 |--- 猫/  
 |--- 狗/
 |--- 鼠/
```
`ImageFolder`可以自动为我们数据打上`0,1,2`这样的标签，并返回图片的路径。给一个示例

```py
from torchvision.datasets import ImageFolder
data = ImageFolder('/data/dogcat/')
print(x.imgs)
# [('/data/dogcat/cat/1.jpg', 0),
#  ('/data/dogcat/cat/2.jpg', 0),
#  ('/data/dogcat/dog/1.jpg', 1),
#  ('/data/dogcat/dog/2.jpg', 1)]
print(x.classes)
# ['cat', 'dog']
```

我们的人脸数据也是以文件夹组织的，真好！

`DataLoader`是另一个非常有用的类，它可以帮助我们生成指定`Batch`大小的数据，并提供打乱数据，快速加载数据的功能。

我们继续在`dataset.py`中加入以下代码:

```py
from config import Config as conf

def load_data(conf, training=True):
    if training:
        dataroot = conf.train_root
        transform = conf.train_transform
        batch_size = conf.train_batch_size
    else:
        dataroot = conf.test_root
        transform = conf.test_transform
        batch_size = conf.test_batch_size

    data = ImageFolder(dataroot, transform=transform)
    class_num = len(data.classes)
    loader = DataLoader(data, batch_size=batch_size, shuffle=True, 
        pin_memory=conf.pin_memory, num_workers=conf.num_workers)
    return loader, class_num
```

这段代码做了以下的事情：

+ 判断是训练还是测试阶段，使用`conf`中的不同参数
+ 生成一个`ImageFolder`对象，而且对数据进行了`transform`，得到`data`
+ 将`data`传入`DataLoader`，指定每次迭代时返回的`batch_size`和其他参数

注意到，有几个参数是我们尚未在`config.py`中指定的，其中`pin_memroy`和`num_workers`都可以来用加速数据加载的过程。

我们现在把这些参数写入`config.py`

```py
class Config:
    # ... 省略 ...
    train_batch_size = 64
    test_batch_size = 60

    pin_memory = True  # if memory is large, set it True for speed
    num_workers = 4  # dataloader
```

这样，数据预处理完成！

### 4 >> 训练参数

回顾我们的流水线：
```py
读取数据 -> 数据预处理 -> 模型得到数据，输出embedding -> embedding和标签进入度量函数，输出概率值 
-> 损失函数计算损失 -> 反向传播调整参数 -> 直到模型收敛
```

现在我们来设置模型参数，在`config.py`中添加

```py
class Config:
    # ... 省略 ...
    backbone = 'fmobile' # [resnet, fmobile]
    metric = 'arcface'  # [cosface, arcface]
    embedding_size = 512
    drop_ratio = 0.5
```
我们指定了使用的网络结构、度量函数、输出人脸特征向量的大小等参数。

接着添加
```py
class Config:
    # ... 省略 ...
    epoch = 30
    optimizer = 'sgd'  # ['sgd', 'adam']
    lr = 1e-1
    lr_step = 10
    lr_decay = 0.95
    weight_decay = 5e-4
    loss = 'focal_loss' # ['focal_loss', 'cross_entropy']
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    checkpoints = "checkpoints"
```
这里指定了学习率，优化方法、损失函数以及设备。`checkpoints`是用来存放权重文件的文件夹。

### 5 >> 训练流水线

配置已经准备好了，我们开始写训练流水线。

创建`train.py`，导入依赖
```py
import os
import os.path as osp

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from model import FaceMobileNet
from model.metric import ArcFace, CosFace
from model.loss import FocalLoss
from dataset import load_data
from config import Config as conf
```

数据设置
```py
# Data Setup
dataloader, class_num = load_data(conf, training=True)
embedding_size = conf.embedding_size
device = conf.device
```

模型及度量函数设置
```py
# Network Setup
net = FaceMobileNet(embedding_size).to(device)

if conf.metric == 'arcface':
    metric = ArcFace(embedding_size, class_num).to(device)
else:
    metric = CosFace(embedding_size, class_num).to(device)

net = nn.DataParallel(net)
metric = nn.DataParallel(metric)
```

损失函数以及优化方法配置
```py
# Training Setup
if conf.loss == 'focal_loss':
    criterion = FocalLoss(gamma=2)
else:
    criterion = nn.CrossEntropyLoss()

if conf.optimizer == 'sgd':
    optimizer = optim.SGD([{'params': net.parameters()}, {'params': metric.parameters()}], 
                            lr=conf.lr, weight_decay=conf.weight_decay)
else:
    optimizer = optim.Adam([{'params': net.parameters()}, {'params': metric.parameters()}],
                            lr=conf.lr, weight_decay=conf.weight_decay)

scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=conf.lr_step, gamma=0.1)
```

创建权重文件夹
```py
# Checkpoints Setup
os.makedirs(conf.checkpoints, exist_ok=True)
```

开始训练啦！

```py
# Start training
net.train()

for e in range(conf.epoch):
    for data, labels in tqdm(dataloader, desc=f"Epoch {e}/{conf.epoch}",
                             ascii=True, total=len(dataloader)):
        data = data.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        embeddings = net(data)
        thetas = metric(embeddings, labels)
        loss = criterion(thetas, labels)
        loss.backward()
        optimizer.step()

    print(f"Epoch {e}/{conf.epoch}, Loss: {loss}")

    backbone_path = osp.join(checkpoints, f"{e}.pth")
    torch.save(net.state_dict(), backbone_path)
    scheduler.step()
```

上面的代码做了以下的事情：

+ 设置网络训练模式
+ 进入一次 epoch 训练，完成之后打印损失
+ 保存一次 epoch 的权重
+ 进行学习率调度

你需要一台GPU机器来完成训练，因为数据量还是挺大的。我在 `WebFace` 上训练时，损失降到 6 的时候，在 LFW 上有 96% 的准确率，这时模型基本上就收敛了。如果换用代码中的`resnet`模型，还可以降到 1 左右，有 97% 左右的准确率。

### 6 >> 资源

+ [config.py](../config.py)
+ [dataset.py](../dataset.py)
+ [train.py](../train.py)

### 7 >> 小结

你已经完成了训练过程，还剩下

+ 006 [测试](./test.md)

### 8 >> 

愿凡有所得，皆能自利利他。
