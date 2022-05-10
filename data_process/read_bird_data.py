# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name:     read_bird_data
   Description:
   Author:        zhangluyao
   date:          2022/5/10
-------------------------------------------------
"""
# *_*coding: utf-8 *_*
# author --liming--

"""
用于已下载数据集的转换,便于pytorch的读取
"""

import torch
import torchvision
import config
from torchvision import datasets, transforms

data_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def train_data_load():
    # 训练集
    root_train = config.ROOT_TRAIN
    train_dataset = torchvision.datasets.ImageFolder(root_train,
                                                     transform=data_transform)
    CLASS = train_dataset.class_to_idx
    print('训练数据label与文件名的关系:', CLASS)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=config.BATCH_SIZE,
                                               shuffle=True)
    return CLASS, train_loader


def test_data_load():
    # 测试集
    root_test = config.ROOT_TEST
    test_dataset = torchvision.datasets.ImageFolder(root_test,
                                                    transform=data_transform)

    CLASS = test_dataset.class_to_idx
    print('测试数据label与文件名的关系：', CLASS)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=config.BATCH_SIZE,
                                              shuffle=True)
    return CLASS, test_loader


if __name__ == '__main___':
    train_data_load()
    test_data_load()