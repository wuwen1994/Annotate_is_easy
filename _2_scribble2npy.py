import os
import numpy as np
import cv2 as cv
from glob import glob
from utils import save_json


def scri2label(paths, Params):
    for path in paths:
        name = path.split('\\')[-1].split(".")[0]
        image = cv.imread(path, -1)
        hight, width = image.shape[0:2]
        mask = np.zeros((hight, width), dtype=np.uint8)
        number_of_shadow = 0
        number_of_non_shadow = 0
        number_of_free = 0
        number_of_all_pixel = hight * width
        for h in range(hight):
            for w in range(width):
                r = image[h, w, 2]
                g = image[h, w, 1]
                b = image[h, w, 0]
                if r >= 250 and g <= 10 and b <= 10:
                    mask[h, w] = 1  # 1为已经确定的阴影区域
                    number_of_shadow += 1
                elif r <= 10 and g >= 250 and b <= 10:
                    mask[h, w] = 0  # 0为已经确定的非阴影区域
                    number_of_non_shadow += 1
                else:
                    mask[h, w] = 2  # 2为不确定区域
                    number_of_free += 1
        save_path = Params['tar_dir'] + name
        np.save(save_path, mask)
        save_path = Params['tar_dir'] + name + '.json'
        save_json(save_path, mask)
        print("被交互到的阴影像素:%s/%s" % (number_of_shadow, number_of_all_pixel))
        print("被交互到的非阴影像素:%s/%s" % (number_of_non_shadow, number_of_all_pixel))
        print("没有被交互到的像素:%s/%s" % (number_of_free, number_of_all_pixel))


if __name__ == '__main__':
    """
    读取scribble图像,白色区域为正例阴影区域,灰色区域为负例非阴影区域,黑色不确定区域，等待传播
    读取scribble图像，给每个像素分配标签，分别使用0，1，2表示
    """
    Params = {
        'root': './data/scribble/',
        'tar_dir': './data/scribble_np/',
    }
    if not os.path.exists(Params['tar_dir']):
        os.makedirs(Params['tar_dir'])
    paths = sorted(glob(Params['root'] + "*.jpg"))
    scri2label(paths)
