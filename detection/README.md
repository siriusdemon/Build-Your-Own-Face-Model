# Build Your Own Face Detection Model

这是一个 CenterFace 的极简实现。

### 依赖

```py
Python >= 3.6
pytorch >= 1.0
torchvision
pillow == 6.2.0
tqdm
```

### 训练

```py
export CUDA_VISIBLE_DEVICES=0
python train.py
```

### 推断

```py
python api.py
```

### 博客

+ 000 [检测原理](./blog/theory.md)
+ 001 [数据准备](./blog/data.md)
+ 002 [数据预处理](./blog/preprocess.md)
+ 003 [模型架构](./blog/model.md)
+ 004 [损失函数](./blog/loss.md)
+ 005 [训练](./blog/train.md)
+ 006 [测试](./blog/test.md)

### 训练你自己的数据

由于这边的格式是`retinaface`的

```sh
# 图片名
left top width height lm1_x lm2_y .... 
left top width height lm1_x lm2_y .... 
```

所以，如果你想训练一个普通的检测网络，你可以把你的标签组织成

```sh
# 图片名
left top width height
left top width height
```

然后，把`datasets.py`中涉及 landmarks 的函数调用，标签生成都去掉就可以了。具体来说，可以从以下两个函数调用入手。

```py
im, bboxes, landmarks = self.preprocess(im, anns)
hm = self.make_heatmaps(im, bboxes, landmarks, self.downscale)
```

### Wishes

愿凡有所得，皆能自利利他。