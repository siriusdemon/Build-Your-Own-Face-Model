# 损失函数

欢迎阅读！本文是 Build Your Own Face Recognition Model 系列博客的第三篇。

在这一节中，我们将设计（FuZhi）一个Focal Loss！

### 1 >> 开始之前

先了解下 Focal Loss 的功能：降低容易样本对 loss 的贡献度，使模型关注那些困难样本。由于简单的样本一般占多数，困难样本占少数，Focal Loss 的这种特点可以使模型学习到更加好的特征。

Focal Loss 是一种思想，而不拘泥于其实现的形式，所以有各种各样的 Focal Loss 实现。

### 2 >> 实现 Focal Loss

在文件夹`model/`下再创建一个文件`loss.py`，写入以下代码： 

```py
import torch
import torch.nn as nn

class FocalLoss(nn.Module):

    def __init__(self, gamma=2):
        super().__init__()
        self.gamma = gamma
        self.ce = torch.nn.CrossEntropyLoss()

    def forward(self, input, target):
        logp = self.ce(input, target)
        p = torch.exp(-logp)
        loss = (1 - p) ** self.gamma * logp
        return loss.mean()
```
关注`forward`过程，`p`可以理解成分类正确的概率。对于容易的样本，`p`值会比较大，因此`(1-p)`接近0，也就是说，这些容易分类的样本对 loss 的贡献变小了。这是核心的思想。`gamma=2`是提出`Focal Loss`的论文`RetinaNet`的经验之谈。当`gamma=0`时，也就褪化为普通的`CrossEntropyLoss`。


### 3 >> 资源

+ [loss.py](../model/loss.py)

### 4 >> 小结

你已经完成了损失函数，还剩下

+ 004 [度量函数](./metric.md)
+ 005 [训练](./train.md)
+ 006 [测试](./test.md)

### 5 >> 

愿凡有所得，皆能自利利他。