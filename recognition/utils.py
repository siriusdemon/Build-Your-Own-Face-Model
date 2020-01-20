"""Train List 训练列表
格式：
ImagePath Label

示例:
/data/WebFace/0124920/003.jpg 10572
/data/WebFace/0124920/012.jpg 10572
/data/WebFace/0124920/020.jpg 10572
"""

import os
import os.path as osp
from imutils import paths

def generate_list(images_directory, saved_name=None):
    """生成数据列表
    Args:
        images_directory: 人脸数据目录，通常包含多个子文件夹。如
            WebFace和LFW的格式
    Returns:
        data_list: [<路径> <标签>]
    """
    subdirs = os.listdir(images_directory)
    num_ids = len(subdirs)
    data_list = []
    for i in range(num_ids):
        subdir = osp.join(images_directory, subdirs[i])
        files = os.listdir(subdir)
        paths = [osp.join(subdir, file) for file in files]
        # 添加ID作为其人脸标签
        paths_with_Id = [f"{p} {i}\n" for p in paths]
        data_list.extend(paths_with_Id)
    
    if saved_name:
        with open(saved_name, 'w', encoding='utf-8') as f:
            f.writelines(data_list)
    return data_list

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

def remove_dirty_image(webface_directory, cleaned_list):
    cleaned_list = set([c.split()[0] for c in cleaned_list])
    for p in paths.list_images(webface_directory):
        if p not in cleaned_list:
            print(f"remove {p}")
            os.remove(p)

if __name__ == '__main__':
    data = '/data/CASIA-WebFace/'
    lst = '/data/cleaned_list.txt'
    cleaned_list = transform_clean_list(data, lst)
    remove_dirty_image(data, cleaned_list)