# 数据准备

欢迎阅读！本文是 Build Your Own Face Recognition Model 系列博客的第一篇。

在这一节，让我们先下载好训练所需要的人脸数据，包括 CASIA-WebFace 以及 LFW 。

### 1 >> 开始之前
```sh
pip install imutils
```
`imutils`是一个图片相关的工具包，我们将会使用它的路径功能。

### 2 >> 下载CASIA-WebFace

CASIA-WebFace 数据集只支持学术机构的免费使用，因此，个人开发者想使用这个数据集有两个办法：

+ 看看其他使用 CASIA-WebFace 的 Github 项目，有没有放出相关的下载链接
+ （推荐）到国内资源站搜索看看，比如百度云，或者CSDN等等


### 3 >> 清洗CASIA-WebFace

CASIA-WebFace 未经清洗前有 4GB 左右的图片，有人对其中不好的图片进行了清洗，整理出了一个干净图片列表（cleaned_list.txt），下载资源附于文末。

在开始训练之前，我将根据干净的图片列表，来删除 CASIA-WebFace 中的脏数据。

`cleaned_list.txt`的内容像这样:
```sh
0000045\001.jpg 0
0000045\002.jpg 0
0000045\003.jpg 0
0000045\004.jpg 0
0000045\005.jpg 0
```

由于我在 linux 上工作，我想把`\`换成`/`，并在路径前添加其文件夹位置，以形成绝对路径。创建一个`utils.py`文件，加入以下代码

```py
import os
import os.path as osp
from imutils import paths


def transform_clean_list(webface_directory, cleaned_list_path):
    """转换webface的干净列表格式
    Args:
        webface_directory: WebFace数据目录
        cleaned_list_path: cleaned_list.txt路径
    Returns:
        cleaned_list: 转换后的数据列表
    """
    with open(cleaned_list_path, encoding='utf-8') as f:
        cleaned_list = f.readlines()
    cleaned_list = [p.replace('\\', '/') for p in cleaned_list]
    cleaned_list = [osp.join(webface_directory, p) for p in cleaned_list]
    return cleaned_list


if __name__ == '__main__':
    data = '/data/CASIA-WebFace/'
    lst = '/data/cleaned_list.txt'
    cleaned_list = transform_clean_list(data, lst)
```


现在列表是这样子的
```sh
/data/CASIA-WebFace/0000045/001.jpg 0
/data/CASIA-WebFace/0000045/002.jpg 0
/data/CASIA-WebFace/0000045/003.jpg 0
/data/CASIA-WebFace/0000045/004.jpg 0
/data/CASIA-WebFace/0000045/005.jpg 0
```
你可以看出，我把数据放到了 `/data`下面，这是我个人的习惯做法。

现在，我准备删除那些脏图片了。在`utils.py`中添加以下代码

```py
def remove_dirty_image(webface_directory, cleaned_list):
    cleaned_list = set([c.split()[0] for c in cleaned_list])
    for p in paths.list_images(webface_directory):
        if p not in cleaned_list:
            print(f"remove {p}")
            os.remove(p)
```
我们使用`imutils.paths`的列出图片的功能，列出所有在`webface_directory`中的图片路径，然后检查这个路径是否在干净列表中，如果不在，我们就删除这张图片。

在`utils.py`的最后，我们添加一行代码。
```py
if __name__ == '__main__':
    data = '/data/CASIA-WebFace/'
    lst = '/data/cleaned_list.txt'
    cleaned_list = transform_clean_list(data, lst)
    remove_dirty_image(data, cleaned_list)   # <- 新增
```

这时候，运行`python3 utils.py`，就开始删除脏图片了。这样训练数据就准备好了。

### 4 >> 下载LFW

我们采用 CASIA-WebFace 作为训练数据，训练之后的模型，跑在 LFW 数据集上来测量模型的效果。下载链接见文末。

下载资源会得到`lfw-align-128.tar.gz`以及`lfw_test_pair.txt`，我同样将之放在 `/data`下，并解压图片
```sh
cd /data
tar -xvf lfw-align-128.tar.gz
rm lfw-align-128.tar.gz
ls
# lfw-align-128  lfw_test_pair.txt  CASIA-WebFace  cleaned list.txt
```

现在，测试数据也准备好了。

### 5 >> 资源

+ [cleaned_list.txt](http://pan.baidu.com/s/1hrKpbm8)
+ [utils.py](../utils.py)
+ [LFW](https://pan.baidu.com/s/12IKEpvM8-tYgSaUiz_adGA) 提取码 u7z4
+ [LFW测试列表](https://github.com/ronghuaiyang/arcface-pytorch/blob/master/lfw_test_pair.txt)

### 6 >> 小结

你已经完成了数据准备，还剩下

+ 002 [模型架构](./model.md)
+ 003 [损失函数](./loss.md)
+ 004 [度量函数](./metric.md)
+ 005 [训练](./train.md)
+ 006 [测试](./test.md)

### 7 >> 

愿凡有所得，皆能自利利他。