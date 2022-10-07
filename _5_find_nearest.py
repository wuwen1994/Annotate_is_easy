# 输入：超像素的特征
# 每个超像素与其他超像素之间的空间相似性
from glob import glob
import numpy as np
from utils import save_json
import os
import cv2 as cv
import copy
import heapq


def spatial_similarity(path,Params):
    for p in path:
        filename = p.split("\\")[-1].split(".")[0]
        features = np.load(p)
        spatial_simi = []
        for i in range(len(features)):
            simi = []
            node1_y = features[i][3]
            node1_x = features[i][4]
            for j in range(len(features)):
                node2_y = features[j][3]
                node2_x = features[j][4]
                y_diff = node2_y - node1_y
                x_diff = node2_x - node1_x
                L2_distance = np.sqrt(y_diff * y_diff + x_diff * x_diff)
                # print(L2_distance)
                simi.append(L2_distance)
            spatial_simi.append(simi)
        near_list = []
        # 6个最近的节点
        for i in range(len(features)):
            tmp = zip(range(len(spatial_simi[i])), spatial_simi[i])
            small7 = heapq.nsmallest(7, tmp, key=lambda x:x[1])
            near = []
            for i in small7:
                near.append(i[0])
            near_list.append(near)
        # 6个随机节点
        for i in range(len(features)):
            count = 0
            while 1:
                if count == 6:
                    break
                index = int(np.random.rand()*len(features))
                if index not in near_list[i]:
                    near_list[i].append(index)
                    count += 1
        near_array = np.array(near_list)
        np.save(Params["connectivity"] + filename, near_array)
        save_json(Params["connectivity"] + filename + ".json", near_array)


if __name__ == '__main__':
    Params = {
        'features': './data/features/',  # 所有超像素的特征
        'connectivity': './data/connectivity/'  # 所有超像素之间的空间相似性
    }
    if not os.path.exists(Params['connectivity']):
        os.makedirs(Params['connectivity'])
    path = sorted(glob(Params['features'] + "*.npy"))
    spatial_similarity(path)
