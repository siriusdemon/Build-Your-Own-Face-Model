# 模型架构

欢迎阅读！本文是 Build Your Own Face Recognition Model 系列博客的第二篇。

在这一节中，我们将只关注模型架构本身！

### 1 >> 开始之前

由于本教程使用Pytorch，先安装依赖
```sh
conda install pytorch torchvision -c pytorch
```

### 2 >> 结构设计

我参考了两三个 Github 仓库的实现，然后按自己比较喜欢的风格写的网络结构。我在`model`文件夹下提供了两种架构，但这里只讲解其中一个。你也可以实现自己的网络结构。

首先，创建一个文件夹叫`model`，创建一个空文件`__init__.py`，再创建一个文件`fmobilenet.py`。

复杂的网络结构可以由一些很简单的块（`block`）来搭建而成，我会先创建一系列这样的块。让我们开始吧！

#### 2.1 >> 块的设计

打开`fmobilenet.py`，先导入依赖
```py
import torch
import torch.nn as nn
import torch.nn.functional as F
```

我们的第一个块叫`Flatten`，它的功能是将一个张量展平
```py
class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.shape[0], -1)
```
`Flatten`用于一系列的卷积操作之后，全连接层之前。因为深度学习模型的卷积操作往往在四维空间里，全连接层在二维空间里，`Flatten`能够将四维空间平铺成二维空间。

第二个块叫`ConvBn`，是卷积操作加一个BN层。
```py
class ConvBn(nn.Module):

    def __init__(self, in_c, out_c, kernel=(1, 1), stride=1, padding=0, groups=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_c)
        )
        
    def forward(self, x):
        return self.net(x)
```

第三个块叫`ConvBnPrelu`，我们在刚刚搭建好的`ConvBn`块里又加了一个`PReLu`激活层。
```py
class ConvBnPrelu(nn.Module):

    def __init__(self, in_c, out_c, kernel=(1, 1), stride=1, padding=0, groups=1):
        super().__init__()
        self.net = nn.Sequential(
            ConvBn(in_c, out_c, kernel, stride, padding, groups),
            nn.PReLU(out_c)
        )

    def forward(self, x):
        return self.net(x)
```

第四个块叫`DepthWise`，`DepthWise`层使用了上面定义的`ConvBnPrelu`和`ConvBn`，它按通道进行卷积操作，实现了高效计算。注意中间的`ConvBnPrelu`块的`groups=groups`。

```py
class DepthWise(nn.Module):

    def __init__(self, in_c, out_c, kernel=(3, 3), stride=2, padding=1, groups=1):
        super().__init__()
        self.net = nn.Sequential(
            ConvBnPrelu(in_c, groups, kernel=(1, 1), stride=1, padding=0),
            ConvBnPrelu(groups, groups, kernel=kernel, stride=stride, padding=padding，groups=groups),
            ConvBn(groups, out_c, kernel=(1, 1), stride=1, padding=0),
        )

    def forward(self, x):
        return self.net(x)
```

第五个块叫`DepthWiseRes`，在第四个块的基础上，添加了一个原始输入，这是`ResNet`系列的精要。
```py
class DepthWiseRes(nn.Module):
    """DepthWise with Residual"""

    def __init__(self, in_c, out_c, kernel=(3, 3), stride=2, padding=1, groups=1):
        super().__init__()
        self.net = DepthWise(in_c, out_c, kernel, stride, padding, groups)

    def forward(self, x):
        return self.net(x) + x
```

第六个块叫`MultiDepthWiseRes`，与之前的不同，我多传入一个`num_block`的参数，由这个参数决定要堆多少个`DepthWiseRes`。由于这些`DepthWiseRes`的输入输出的通道数是一样的，所以堆多少都不会引起通道数的变化。

```py
class MultiDepthWiseRes(nn.Module):

    def __init__(self, num_block, channels, kernel=(3, 3), stride=1, padding=1, groups=1):
        super().__init__()

        self.net = nn.Sequential(*[
            DepthWiseRes(channels, channels, kernel, stride, padding, groups)
            for _ in range(num_block)
        ])

    def forward(self, x):
        return self.net(x)
```

至此，我们完成了六种块的设计，现在我们用这些块来搭建我们的网络结构！

#### 2.2 >> 网络结构

继续往`fmobilenet.py`添加代码

```py
class FaceMobileNet(nn.Module):

    def __init__(self, embedding_size):
        super().__init__()
        self.conv1 = ConvBnPrelu(1, 64, kernel=(3, 3), stride=2, padding=1)
        self.conv2 = ConvBn(64, 64, kernel=(3, 3), stride=1, padding=1, groups=64)
        self.conv3 = DepthWise(64, 64, kernel=(3, 3), stride=2, padding=1, groups=128)
        self.conv4 = MultiDepthWiseRes(num_block=4, channels=64, kernel=3, stride=1, padding=1, groups=128)
        self.conv5 = DepthWise(64, 128, kernel=(3, 3), stride=2, padding=1, groups=256)
        self.conv6 = MultiDepthWiseRes(num_block=6, channels=128, kernel=(3, 3), stride=1, padding=1, groups=256)
        self.conv7 = DepthWise(128, 128, kernel=(3, 3), stride=2, padding=1, groups=512)
        self.conv8 = MultiDepthWiseRes(num_block=2, channels=128, kernel=(3, 3), stride=1, padding=1, groups=256)
        self.conv9 = ConvBnPrelu(128, 512, kernel=(1, 1))
        self.conv10 = ConvBn(512, 512, groups=512, kernel=(7, 7))
        self.flatten = Flatten()
        self.linear = nn.Linear(2048, embedding_size, bias=False)
        self.bn = nn.BatchNorm1d(embedding_size)
        
    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)
        out = self.conv6(out)
        out = self.conv7(out)
        out = self.conv8(out)
        out = self.conv9(out)
        out = self.conv10(out)
        out = self.flatten(out)
        out = self.linear(out)
        out = self.bn(out)
        return out
```
由于前面已经定义好了六个块，现在这个`FaceMobilenet`的结构一目了然。我们先是堆叠了10种不同的卷积块，然后接一个`Flatten`块把输入展平，再接一个全连接层和1维的`BatchNorm`层。值得说一下的是这一行代码

```py
class FaceMobilenet(nn.Module):
        # ... emit ...
        self.linear = nn.Linear(2048, embedding_size, bias=False)
```
由于我们的输入是`1 x 128 x 128`，经过多层卷积之后，其变成`512 x 2 x 2`，也就是`2048`。如果你不知道或者你懒得去算输入的图片经过卷积之后的维度是多少，你可以给网络传入一个假数据，报错信息会告诉你这个维度的值。

另外，这里的`embedding_size`由外部传入，它表示用多大的向量来表示一张人脸。像 Facenet 是使用了128维的向量来表征一张人脸，我们这里使用512。

至此，我们的网络结构就设计完毕啦！

### 3 >> 测试

用假数据测试网络是有必要的，这往往可以帮助我们发现维度匹配的问题。我们继续在`fmobilenet.py`里添加以下代码

```py
if __name__ == "__main__":
    from PIL import Image
    import numpy as np

    x = Image.open("../samples/009.jpg").convert('L')
    x = x.resize((128, 128))
    x = np.asarray(x, dtype=np.float32)
    x = x[None, None, ...]
    x = torch.from_numpy(x)
    net = FaceMobileNet(512)
    net.eval()
    with torch.no_grad():
        out = net(x)
    print(out.shape)
```

这里打开的图片路径必须是存在的哦！保存后，在命令行中运行
```sh
python3 fmobilenet.py
# => torch.Size([1, 512])
```

### 4 >> 资源

+ [fmobilenet.py](../model/fmobilenet.py)


### 5 >> 小结

你已经完成了模型架构，还剩下

+ 003 [损失函数](./loss.md)
+ 004 [度量函数](./metric.md)
+ 005 [训练](./train.md)
+ 006 [测试](./test.md)

### 6 >> 

愿凡有所得，皆能自利利他。