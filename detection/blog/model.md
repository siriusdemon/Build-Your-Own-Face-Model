# 网络架构

本篇是 Build Your Own Face Detection Model 的第四节。

这一节，我们将探索网络的架构。

### 1 >> 开始之前

这一节不写代码。嗯，因为上一节写太多了。

### 2 >> 谈谈架构

CenterFace 的骨干属于 encoder-decoder 架构，跟分割网络很接近，比如 Unet，FCN 之类的。但它有个限制，就是最后输出的 heatmap 的边的大小应该是原来的 1/4。如果输入是 800x800，则输出是 200x200。这个我们上一节已经讲过了。

不过，这个限制也并不是不可以改动的，只要你有足够的显卡。我想 Cornernet 是 2017 年的论文，现在已经是 2020 年了。17 年还在想怎么节约内存，20 年已经是一股脑用上各种 trick 的时代了。

顺便强调一下，本系列实现的 CenterFace，基本没有 trick 。

### 3 >> 模型代码

模型代码我直接拿了别人写好的，原文件应该是类似[这个](https://github.com/chenjun2hao/CenterFace.pytorch/blob/master/src/lib/models/Backbone/centerface_mobilenet_v2.py)，但我不确定是不是，因为同个目录下还有个带`fpn`的。

因为骨干网络是最好替换的，所以嘛！

创建`models`文件夹，创建一个`mnet.py`文件，把代码复制进去。最后，加一个`__init__.py`。

### 4 >> 

愿凡有所得，皆能自利利他。