# 测试

欢迎阅读！本文是 Build Your Own Face Recognition Model 系列博客的第六篇。

在这一节，我们将测试我们上一节所训练的模型！

### 1 >> 开始之前

我们在训练的时候，是训练模型完成正确的分类。在测试阶段，模型将计算两张人脸的相似度。所以我们的测试列表`lfw_test_pair.txt`是这样子的

```sh
Abel_Pacheco/Abel_Pacheco_0001.jpg Abel_Pacheco/Abel_Pacheco_0004.jpg 1
Akhmed_Zakayev/Akhmed_Zakayev_0001.jpg Akhmed_Zakayev/Akhmed_Zakayev_0003.jpg 1
... ...
Enrique_Iglesias/Enrique_Iglesias_0001.jpg Gisele_Bundchen/Gisele_Bundchen_0002.jpg 0
Eric_Bana/Eric_Bana_0001.jpg Mike_Sweeney/Mike_Sweeney_0001.jpg 0
```

即`人脸1 人脸2 [标签]`，1表示同一个人，0表示不同。我们采用第一篇就准备好的 LFW 数据集来测试。`lfw_test_pair.txt`提供了3000对正例，和3000对反例，共6000个测试样例。测试流水线如下： 
```py
读取样本 -> 分组(batch) -> 模型计算 embeddings -> 保存字典 {imgPath: embeddings} -> 计算人脸划分的阈值和准确度
```

### 2 >> 读取样本

创建一个`test.py`，加入以下代码

```py
import os
import os.path as osp

import torch
import torch.nn as nn
import numpy as np
from PIL import Image

from config import Config as conf
from model import FaceMobileNet
```

由于6000个测试用例中，图片是有重复的。我先获取每一个不重复图片的路径。为了不重复，用集合就可以了！
```py
def unique_image(pair_list) -> set:
    """Return unique image path in pair_list.txt"""
    with open(pair_list, 'r') as fd:
        pairs = fd.readlines()
    unique = set()
    for pair in pairs:
        id1, id2, _ = pair.split()
        unique.add(id1)
        unique.add(id2)
    return unique
```

### 3 >> 分组

经过上一步，得到 7000 多个图片，我们一批一批地计算它们的 embeddings，也称为特征，后文将用特征代表 embeddings。这里我打算自己写分组函数，也让读者对于整个数据处理过程有个更清晰的认识。

```py
def group_image(images: set, batch) -> list:
    """Group image paths by batch size"""
    images = list(images)
    size = len(images)
    res = []
    for i in range(0, size, batch):
        end = min(batch + i, size)
        res.append(images[i : end])
    return res
```

### 4 >> 数据预处理

分组好了之后，进行数据预处理。函数名以下划线开头，表示这个函数不希望被用户直接使用。

```py
def _preprocess(images: list, transform) -> torch.Tensor:
    res = []
    for img in images:
        im = Image.open(img)
        im = transform(im)
        res.append(im)
    data = torch.cat(res, dim=0)  # shape: (batch, 128, 128)
    data = data[:, None, :, :]    # shape: (batch, 1, 128, 128)
    return data
```

### 5 >> 计算特征

计算一批图片的特征，并返回一个特征字典。

```py
def featurize(images: list, transform, net, device) -> dict:
    """featurize each image and save into a dictionary
    Args:
        images: image paths
        transform: test transform
        net: pretrained model
        device: cpu or cuda
    Returns:
        Dict (key: imagePath, value: feature)
    """
    data = _preprocess(images, transform)
    data = data.to(device)
    net = net.to(device)
    with torch.no_grad():
        features = net(data) 
    res = {img: feature for (img, feature) in zip(images, features)}
    return res
```

### 6 >> 余弦距离

我采用余弦距离来度量两张人脸的距离，这跟训练过程是对应的。

```py
def cosin_metric(x1, x2):
    return np.dot(x1, x2) / (np.linalg.norm(x1) * np.linalg.norm(x2))
```

### 7 >> 人脸划分

当 A、B 两张人脸的距离多大时，才认为 A、B 是不同的人？ 以下的代码纯属复制，版权归原作者。
```py

def threshold_search(y_score, y_true):
    y_score = np.asarray(y_score)
    y_true = np.asarray(y_true)
    best_acc = 0
    best_th = 0
    for i in range(len(y_score)):
        th = y_score[i]
        y_test = (y_score >= th)
        acc = np.mean((y_test == y_true).astype(int))
        if acc > best_acc:
            best_acc = acc
            best_th = th
    return best_acc, best_th
```

他做了以下的事情：

+ 取一对人脸的距离作阈值
+ 用选好的阈值进行划分，大于此阈值的是同一个人的脸，小于此阈值的是不同人的脸
+ 计算这样子划分的准确率
+ 取下一个人脸的距离作为阈值，直到遍历完成
+ 返回最佳准确率以阈值


### 8 >> 计算准确率

```py
def compute_accuracy(feature_dict, pair_list, test_root):
    with open(pair_list, 'r') as f:
        pairs = f.readlines()

    similarities = []
    labels = []
    for pair in pairs:
        img1, img2, label = pair.split()
        img1 = osp.join(test_root, img1)
        img2 = osp.join(test_root, img2)
        feature1 = feature_dict[img1].cpu().numpy()
        feature2 = feature_dict[img2].cpu().numpy()
        label = int(label)

        similarity = cosin_metric(feature1, feature2)
        similarities.append(similarity)
        labels.append(label)

    accuracy, threshold = threshold_search(similarities, labels)
    return accuracy, threshold
```

### 9 >> 

现在，一切准备就绪。在`test.py`中继续添加以下代码：

```py
if __name__ == '__main__':

    model = FaceMobileNet(conf.embedding_size)
    model = nn.DataParallel(model)
    model.load_state_dict(torch.load(conf.test_model, map_location=conf.device))
    model.eval()

    images = unique_image(conf.test_list)
    images = [osp.join(conf.test_root, img) for img in images]
    groups = group_image(images, conf.test_batch_size)

    feature_dict = dict()
    for group in groups:
        d = featurize(group, conf.test_transform, model, conf.device)
        feature_dict.update(d) 
    accuracy, threshold = compute_accuracy(feature_dict, conf.test_list, conf.test_root) 

    print(
        f"Test Model: {conf.test_model}\n"
        f"Accuracy: {accuracy:.3f}\n"
        f"Threshold: {threshold:.3f}\n"
    )
```

以上代码做了以下的事情：

+ 加载预训练模型，记得在`config.py`中添加`test_model`，指向你想测试的权重文件
+ 进行图片分组
+ 计算人脸特征
+ 进行阈值搜索
+ 打印结果

### 10 >> 资源

+ [test.py](../test.py)

### 11 >> 小结

恭喜你，你已经完成了全部教程！还有什么可以做的呢？

1. 测试阶段增强，比如对图片进行水平翻转，将两个特征合成一个，来作为人脸表示
2. 训练阶段增强，你可以引入一个检测分支，这个分支用于判断输入的图片是否是人脸
3. 数据增强，你可以用人脸关键点检测器检测出人脸的五官，然后交换这些器官的位置（你是不是想起了胶囊网络？），让模型学会判断五官的位置。

以上2、3点是我自己的想法，如果你知道别人已经有这样的实现了，好心告诉我！如果你实践了这种想法并取得成绩，我会很高兴！

### 12 >> 

愿凡有所得，皆能自利利他。
