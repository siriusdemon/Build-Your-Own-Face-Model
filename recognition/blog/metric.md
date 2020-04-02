# 度量函数

欢迎阅读！本文是 Build Your Own Face Recognition Model 系列博客的第四篇。

在这一节，我们将实现两个度量函数，CosFace以及ArcFace。

### 1 >> 开始之前

后 Facenet 时代，研究员觉得三元组的选择太麻烦，于是在原来的 Softmax Loss 基础上进行改进，演绎出了像 SphereFace, CosFace, ArcFace 这样的 Additional Margin Metric Loss 。

Additional Margin Metric Loss 的本质，是使得训练的过程更加困难，通过这种困难，磨砺模型，使其训练得更好。举例来说

假设我们的数据集共有 3 个人，在拿到模型对这 3 个人的概率之后，Softmax 的做法是：使标签对应的概率值是 3 个人中最大的，如下例

```py
Softmax: input = '第3个人.jpg' -> model -> 概率 [0.2, 0.2, 0.7] -> 完成任务
```
Softmax 只要做到，输入是哪个人，哪个人的概率就是最高的，就算完成任务了。

CosFace 那一类的度量函数可不是这么简单的，他们的工作流有点像这样

```py
CosFace: input = '第3个人.jpg' -> model -> 概率 [0.2, 0.2, 0.7] 
                -> 增强训练，第3个人的概率要减掉0.5 -> 概率 [0.2, 0.2, 0.2] -> 未完成，继续训练
CosFace: input = '第3个人.jpg' -> model -> 概率 [0.2, 0.2, 0.9] 
                -> 增强训练，第3个人的概率要减掉0.5 -> 概率 [0.2, 0.2, 0.4] -> 完成任务
```

所以，你看到，像 CosFace 这样的训练完成之后，不同的类别之间，会有一个额外的差距，这就是所谓的 Additional Margin！ SphereFace， CosFace, ArcFace 他们的差别只在于，这个 margin 的位置在哪里而已！

按理说，此处应有公式推导，但公式推导值得另起文章书写，鉴于网络上有大量资源，这里不再重复。

### 2 >> 实现CosFace

在`model/`下创建`metric.py`，写入以下代码

```py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class CosFace(nn.Module):

    def __init__(self, in_features, out_features, s=30.0, m=0.40):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, input, label):
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        phi = cosine - self.m
        output = cosine * 1.0  # make backward works
        batch_size = len(output)
        output[range(batch_size), label] = phi[range(batch_size), label]
        return output * self.s
```
简单解释，`s`是放大的因子，由于输入和权重都被进行了 L2 规范化，他们乘出来的 cosine 值也是在`[-1, 1]`之间，这就使得反向传播的梯度太小了，而没有经过 L2 范化的值域一般在`[-20, 80]`这个区间左右，因此，L2 规范化之后需要进行放大，这部分的解释看原论文更清楚。

总而言之，CosFace做了以下事情：

+ 将 backbone 网络的输出，也就是 embedding 进行 L2 规范化。
+ 将 CosFace 度量函数的权重进行 L2 规范化，这样，与 embedding 的线性相乘即是其 cosine 值。
+ 对正确标签的输出进行强化，也就是减小于概率值。
+ 对强化后的 cosine 进行放大，以便后续的反向传播可以工作。

上面的`forward`函数做了以下事情：

+ 将`input`和`weight`进行规范化，并计算其夹角`cosine`
+ `cosine`减去预先定义好的额外差距`m`，得到`phi`
+ `output = cosine * 1.0`是为了避免直接修改了`cosine`的值，影响`Pytorch`的正常反向传播过程。
+ 最后一行的意思是，将`output`中正确标签的概率值，替换成经过第二步强化的概率值
+ 返回时进行放大

如果你理解了上面的过程，后面理解 ArcFace 就是手到擒来！

### 3 >> 实现ArcFace

继续在`model/metric.py`中添加以下代码

```py
class ArcFace(nn.Module):
    
    def __init__(self, embedding_size, class_num, s=30.0, m=0.50):
        super().__init__()
        self.in_features = embedding_size
        self.out_features = class_num
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(class_num, embedding_size))
        nn.init.xavier_uniform_(self.weight)

        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label):
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        sine = ((1.0 - cosine.pow(2)).clamp(0, 1)).sqrt()
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)  # drop to CosFace
        output = cosine * 1.0  # make backward works
        batch_size = len(output)
        output[range(batch_size), label] = phi[range(batch_size), label]
        return output * self.s
```

ArcFace 看起来多了很多东西，其实是因为它的度量存在越界问题。Arc 代表角，其额外差距`m`是一个角度，而 CosFace 中的`m`是一个余弦值。ArcFace 的越界发生在：原来的角度加上额外的角度超过180度的时候。上面的代码第3、4行说的是，如果越界发生了，就使用 CosFace 代替 ArcFace，所以那些额外的变量和计算过程都是为了完成从角度空间向余弦空间的转换而已。

至此，度量函数完成！学习到 ArcFace 这样的论文时我眼前一亮，因为这些都是符合人类直觉的数学，而不是什么神秘代码！

### 4 >> 完善 model

`model/`下的内容都已经完整了，你应该包含以下文件
```sh
model/
├── __init__.py     # 第2篇
├── fmobilenet.py   # 第2篇
├── loss.py         # 第3篇
└── metric.py       # 第4篇
```

打开`__init__.py`，加入以下代码
```py
from .fmobilenet import FaceMobileNet
from .loss import FocalLoss
from .metric import ArcFace, CosFace
```

### 5 >> 资源

+ [Normface](https://arxiv.org/abs/1704.06369)
+ [metric.py](../model/metric.py)

### 6 >> 小结

你已经完成了度量函数，还剩下

+ 005 [训练](./train.md)
+ 006 [测试](./test.md)

### 6 >> 

愿凡有所得，皆能自利利他。