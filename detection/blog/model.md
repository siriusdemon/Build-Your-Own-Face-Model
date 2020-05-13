# 网络架构

本篇是 Build Your Own Face Detection Model 的第四节。

这一节，我们将探索网络的架构。

### 1 >> 开始之前

这一节不写代码：）。

### 2 >> 骨干网络

CenterFace/Centernet 的骨干属于 encoder-decoder 架构，跟分割网络很接近，比如 Unet，FCN 之类的。但它有个限制，就是最后输出的 heatmap 的边的大小应该是原来的 1/4。如果输入是 800x800，则输出是 200x200。这个我们上一节已经讲过了。

不过，这个限制也并不是不可以改动的，只要你有足够的显卡。在 CenterFace 中，作者说是为了减少内存使用。

### 3 >> 模型代码

模型代码我直接拿了别人写好的，原文件应该是类似[这个](https://github.com/chenjun2hao/CenterFace.pytorch/blob/master/src/lib/models/Backbone/centerface_mobilenet_v2.py)，

因为骨干网络是最好替换的，你可以尝试 Stacked hourglass 以及 Unet++ 这样的架构。不过为了快速收敛，这次我用的是 mobilenet 。

创建`models`文件夹，创建一个`mnet.py`文件，把代码复制进去。最后，加一个`__init__.py`。

具体使用的时候，只调用文件末尾的`get_mobile_net`。比如我想做普通的 Centernet，可以这样调

```py
net = get_mobile_net(10, {'hm':80, 'wh':2, 'off':2}, head_conv=24)
```
由于 COCO 中有 80 类，所以`'hm':80`。但 CenterFace 只关心人脸，所以是

```py
net = get_mobile_net(10, {'hm':1, 'wh':2, 'lm':10, 'off':2}, head_conv=24)
```

多一个`'lm':10`，是因为我们还想检测人脸关键点。

### 4 >> 

愿凡有所得，皆能自利利他。