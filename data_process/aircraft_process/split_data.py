# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name:     split_data
   Description:
   Author:        zhangluyao
   date:          2022/5/10
-------------------------------------------------
"""
import pandas as pd
import os, shutil, time
from tqdm import tqdm


def my_mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


# ===== settings =====
src_dir = r'/Users/jack/Downloads/细粒度分类/fgvc-aircraft-2013b/data'
img_dir = os.path.join(src_dir, 'images')
dst_dir = r'/Users/jack/Downloads/细粒度分类/aircraft'


if __name__ == '__main__':
    begin = time.time()

    for method in ['family', 'manufacturer', 'variant']:
        method_dir = os.path.join(dst_dir, 'fgvc_{}'.format(method))
        my_mkdir(method_dir)

        for dataset in ['train', 'val', 'trainval', 'test']:
            dataset_dir = os.path.join(method_dir, dataset)
            my_mkdir(dataset_dir)
            txt = pd.read_csv(os.path.join(src_dir, 'images_{}_{}.txt'.format(method, dataset)),
                              header=None).to_numpy().flatten()

            for info in tqdm(txt, desc='Copying {} {}'.format(method, dataset)):

                if '/' in info:
                    info = info.replace('/', '_')

                img, cls = info.split(' ', 1)
                cls_dir = os.path.join(dataset_dir, cls)
                my_mkdir(cls_dir)
                shutil.copyfile(os.path.join(img_dir, '{}.jpg'.format(img)),
                                os.path.join(cls_dir, '{}.jpg'.format(img)))

    print('\nAll Done, {} s used.'.format(time.time() - begin))