from glob import glob
import numpy as np
import scipy.sparse as sp
from utils import save_json
import os
import cv2 as cv
import copy
import heapq


def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""

    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx


class Weighting():
    def __init__(self):
        super(Weighting, self).__init__()
        self.description = "颜色特征 + 空间位置特征"
        self.weights1 = []
        self.weights2 = []

    def weights_for(self, idx1, idx2, features):  # feature : (1420, 5)
        # 拿出已经计算好的特征
        rgb1 = features[idx1, 0:3]  # 节点1的三通道像素值
        rgb2 = features[idx2, 0:3]  # 节点2的三通道像素值

        pos1 = features[idx1, 3:5]
        pos2 = features[idx2, 3:5]

        rgb_diff = rgb1 - rgb2  # 颜色差异
        pos_diff = pos1 - pos2  # 空间位置差异

        L2_rgb = np.sum(rgb_diff * rgb_diff)  # 颜色的L2 distance
        L2_pos = np.sum(pos_diff * pos_diff)  # 空间位置的L2 distance

        self.weights1.append(L2_rgb)  # 保存颜色的相似性
        self.weights2.append(L2_pos)  # 保存空间位置相似性

    def post_process(self, args=None):
        self.weights1 = np.asarray(self.weights1, dtype=np.float32)  # 266896
        self.weights2 = np.asarray(self.weights2, dtype=np.float32)  # 266896

        num_nodes = args["num_nodes"]
        ne = float(self.weights1.shape[0])  # number of edges 边的数量

        muw1 = self.weights1.sum() / ne  # weight2的平均值
        muw2 = self.weights2.sum() / ne  # weight3的平均值

        sig1 = 2 * np.sum((self.weights1 - muw1) ** 2) / ne  # 权重2的方差
        sig2 = 2 * np.sum((self.weights2 - muw2) ** 2) / ne  # 权重3的方差

        self.weights1 = np.exp(-self.weights1 / sig1)
        self.weights2 = np.exp(-self.weights2 / sig2)

        w1 = sp.coo_matrix((self.weights1, (args["edges"][:, 0], args["edges"][:, 1])), shape=(num_nodes, num_nodes))
        w2 = sp.coo_matrix((self.weights2, (args["edges"][:, 0], args["edges"][:, 1])), shape=(num_nodes, num_nodes))

        self.weights = w1 + 2 * w2

    def get_weights(self):
        return self.weights


def creating_graph(f_p, c_p, s2l_p,Params):
    number_of_image = len(f_p)
    for i in range(number_of_image):
        p1 = f_p[i]
        p2 = c_p[i]
        p3 = s2l_p[i]
        filename = p1.split("\\")[-1].split(".")[0]
        features = np.load(p1)
        connectivity = np.load(p2)
        s2l = np.load(p3)
        # 特征标准化

        number_of_sp, features_dim = features.shape[:]
        R = features[:, 0]
        G = features[:, 1]
        B = features[:, 2]
        y = features[:, 3]
        x = features[:, 4]

        # 求均值
        mean_r = R.sum() / number_of_sp
        mean_g = G.sum() / number_of_sp
        mean_b = B.sum() / number_of_sp
        mean_y = y.sum() / number_of_sp
        mean_x = x.sum() / number_of_sp
        # 求方差
        sigma_r = np.sum((R - mean_r) ** 2) / number_of_sp
        sigma_g = np.sum((G - mean_g) ** 2) / number_of_sp
        sigma_b = np.sum((B - mean_b) ** 2) / number_of_sp
        sigma_y = np.sum((y - mean_y) ** 2) / number_of_sp
        sigma_x = np.sum((x - mean_x) ** 2) / number_of_sp

        # 去均值除方差
        r_features = (R - mean_r) / sigma_r
        g_features = (G - mean_g) / sigma_g
        b_features = (B - mean_b) / sigma_b
        y_features = (y - mean_y) / sigma_y
        x_features = (x - mean_x) / sigma_x

        # 特征拓展
        new_r = np.expand_dims(r_features, axis=1)
        new_g = np.expand_dims(g_features, axis=1)
        new_b = np.expand_dims(b_features, axis=1)
        new_y = np.expand_dims(y_features, axis=1)
        new_x = np.expand_dims(x_features, axis=1)

        # 特征合并
        features = np.concatenate((new_r, new_g, new_b, new_y, new_x), axis=1)  # 所有节点构成的特征矩阵  (1420, 5)

        # 根据connectivity构建图
        edges = []
        tabu_list = {}  # 防止定义重复边
        weighting = Weighting()
        for node_x in range(number_of_sp):
            # print("nodex", node_x)
            for node_y in range(1, len(connectivity[node_x])):
                if (node_x, connectivity[node_x, node_y]) not in tabu_list and (
                        connectivity[node_x, node_y], node_x) not in tabu_list:  # 判断某两个节点的边是否存在
                    tabu_list[(node_x, connectivity[node_x, node_y])] = 1  # adding the edge to the tabu list
                    weighting.weights_for(node_x, connectivity[node_x, node_y], features)  # 计算权重并保存到权重矩阵中
                    weighting.weights_for(connectivity[node_x, node_y], node_x, features)  # 保存其对称位置的权重
                    edges.append([node_x, connectivity[node_x, node_y]])
                    edges.append([connectivity[node_x, node_y], node_x])
        edges = np.asarray(edges, dtype=int)  # (26814, 2)
        pp_args = {
            "edges": edges,
            "num_nodes": number_of_sp
        }
        weighting.post_process(pp_args)
        weights = weighting.get_weights()
        edges, weights, _ = sparse_to_tuple(weights)

        # 生成uncertainty mask
        uncertain_mask = []
        for value in s2l:
            if value == 2:
                uncertain_mask.append(1)
            else:
                uncertain_mask.append(0)
        uncertain_mask = np.expand_dims(np.array(uncertain_mask), axis=1)

        np.save(Params['graph'] + "graph.npy", edges)
        np.save(Params['graph'] + "graph_weights.npy", weights)
        np.save(Params['graph'] + "graph_node_features.npy", features)
        np.save(Params['graph'] + "graph_ground_truth.npy", s2l)
        np.save(Params['graph'] + "unc_mask.npy", uncertain_mask)  # 不确定性的点，不参与训练的部分。以可靠像素样本为中心的构建的Graph才参与训练

        print("------------构建图的结果-----------")
        print("图的形状: {}".format(edges.shape))
        print("权重的形状: {}".format(weights.shape))
        print("图特征的形状: {}".format(features.shape))  # RGB3通道,期望值的1通道,熵值的1通道
        print("训练样本的标签形状: {}".format(s2l.shape))
        print("不确定性像素mask的形状: {}".format(uncertain_mask.shape))
        print("参与计算的node数量: {}".format(number_of_sp))
        print("没有被注释到的node数量: {}".format(int(np.sum(s2l[s2l == 2] / 2))))
        print("被注释到的node数量: {}".format(number_of_sp - int(np.sum(s2l[s2l == 2] / 2))))
        print("被注释到的阴影node数量: {}".format(int(np.sum(s2l[s2l == 1]))))
        print("被注释到的非阴影node数量: {}".format(number_of_sp - int(np.sum(s2l[s2l == 2] / 2)) - int(np.sum(s2l[s2l == 1]))))



if __name__ == '__main__':
    Params = {
        'features': './data/features/',  # 所有超像素的特征
        'connectivity': './data/connectivity/',  # 所有超像素之间的空间相似性'
        'sp_index2label': "./data/sp_index2label/",
        'graph': './data/graph/'
    }
    if not os.path.exists(Params['graph']):
        os.makedirs(Params['graph'])

    features_path = sorted(glob(Params['features'] + "*.npy"))
    connectivity_path = sorted(glob(Params['connectivity'] + "*.npy"))
    sp_index2label_path = sorted(glob(Params['sp_index2label'] + "*.npy"))
    creating_graph(features_path, connectivity_path, sp_index2label_path)
