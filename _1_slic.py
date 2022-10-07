import os
import cv2 as cv
import numpy as np
from glob import glob
from skimage.segmentation import slic, mark_boundaries
from utils import save_json


def gen_superpixel_label(paths,n_segments, compactness,Params):
    for i, path in enumerate(paths):
        print("----[%3s/%s] superpixel segmentating----" % (i + 1, len(paths)))
        name = path.split('\\')[-1]
        img = cv.imread(path)
        segments = slic(img, n_segments=n_segments, compactness=compactness, start_label=0)  # shape (H, W)

        # segments 是一张与原图等大的矩阵，记录着每个位置属于几号超像素
        save_path = Params['tar_dir'] + name.split('.jpg')[0]
        np.save(save_path, segments)
        save_path = Params['tar_dir'] + name.split('.jpg')[0] + '.json'
        save_json(save_path, segments)
        # 可视化效果
        boundary = (mark_boundaries(img, segments, mode='thick') * 255).astype(np.uint8)
        save_path = Params['tar_dir2'] + name
        cv.imwrite(save_path, boundary)


if __name__ == '__main__':
    Params = {
        'root': './data/image/',
        'tar_dir': './data/superpixel/',
        'tar_dir2': './data/superpixel_vis/'
    }
    if not os.path.exists(Params['tar_dir']):
        os.makedirs(Params['tar_dir'])
    if not os.path.exists(Params['tar_dir2']):
        os.makedirs(Params['tar_dir2'])

    # 定义超像素个数
    n_segments, compactness = 1500, 20
    paths = sorted(glob(Params['root'] + "*.jpg"))
    gen_superpixel_label(paths)
