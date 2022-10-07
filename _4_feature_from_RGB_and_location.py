from glob import glob
import numpy as np
from utils import save_json
import os
import cv2 as cv


def feature_extracting(path1, path2, path3, Params):
    number_of_image = len(path1)
    for i in range(number_of_image):
        p1 = path1[i]
        p2 = path2[i]
        p3 = path3[i]
        filename = p1.split("\\")[-1].split(".")[0]
        image = cv.imread(p1, -1)
        superpixel_mask = np.load(p2)
        sp_index2label = np.load(p3)
        number_of_sp = len(sp_index2label)
        features = []
        for i in range(number_of_sp):
            print("----%s-th superpixel feature extracting----" % i)
            height = np.where(superpixel_mask == i)[0]
            width = np.where(superpixel_mask == i)[1]
            r_mean = 0
            g_mean = 0
            b_mean = 0
            h_mean = 0
            w_mean = 0
            for h, w in zip(height, width):
                r_mean += image[h, w, 2] / len(height)
                g_mean += image[h, w, 1] / len(height)
                b_mean += image[h, w, 0] / len(height)
                h_mean += h / len(height)
                w_mean += w / len(height)
            features.append([r_mean, g_mean, b_mean, h_mean, w_mean])
        features = np.array(features).astype(np.float32)
        np.save(Params["features"] + filename, features)
        save_json(Params["features"] + filename + ".json", features)


if __name__ == '__main__':
    Params = {
        'image': './data/image/',  # 原始图像 用于提供RGB信息和空间位置信息
        'superpixel': './data/superpixel/',  # 超像素mask，用于标识某个超像素在原图中包含哪些像素
        'sp_index2label': './data/sp_index2label/',  # 用于提供有多少个超像素
        'features': "./data/features/"
    }
    if not os.path.exists(Params['features']):
        os.makedirs(Params['features'])
    path1 = sorted(glob(Params['image'] + "*.jpg"))
    path2 = sorted(glob(Params['superpixel'] + "*.npy"))
    path3 = sorted(glob(Params['sp_index2label'] + "*.npy"))
    feature_extracting(path1, path2, path3)
