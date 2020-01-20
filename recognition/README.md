# Build Your Own Face Recognition Model

训练你自己的人脸识别模型！

人脸识别从原始的 Softmax Embbedding，经过2015年 Facenet 领衔的 triple loss metric learning，然后是 additional margin metric learning。这次的系列博客实现的是2018年提出的 ArcFace 。


### 依赖
```py
Python >= 3.6
pytorch >= 1.0
torchvision
imutils
pillow == 6.2.0
tqdm
```

### 数据准备

+ 下载WebFace（百度一下）以及干净的图片列表（[BaiduYun](http://pan.baidu.com/s/1hrKpbm8)）用于训练
+ 下载LFW（[BaiduYun](https://pan.baidu.com/s/12IKEpvM8-tYgSaUiz_adGA) 提取码 u7z4）以及[测试列表](https://github.com/ronghuaiyang/arcface-pytorch/blob/master/lfw_test_pair.txt)用于测试
+ 删除WebFace中的脏数据，使用`utils.py`

### 配置参数

见`config.py`

### 训练

天然支持单机多GPU训练

```py
export CUDA_VISIBLE_DEVICES=0,1
python train.py
```

### 测试

```py
python test.py
```

### 博客

虽然有关人脸识别的介绍已经很多了，但受到许多 [Build-Your-Own-x](https://github.com/danistefanovic/build-your-own-x) 文章的启发，就想写一个 Build Your Own Face Model 的博客，愿于他人有益。

+ 001 [数据准备](./blog/data.md)
+ 002 [模型架构](./blog/model.md)
+ 003 [损失函数](./blog/loss.md)
+ 004 [度量函数](./blog/metric.md)
+ 005 [训练](./blog/train.md)
+ 006 [测试](./blog/test.md)

### 致谢

虽然并未注明，但本项目中有一些代码直接复制或者修改自以下仓库，许可证与之相同：

+ [insightFace](https://github.com/deepinsight/insightface/tree/master/recognition)
+ [insightFace_Pytorch](https://github.com/TreB1eN/InsightFace_Pytorch)
+ [arcface-pytorch](https://github.com/ronghuaiyang/arcface-pytorch)
