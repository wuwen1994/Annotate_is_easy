import os
from glob import glob
import numpy as np
from _1_slic import gen_superpixel_label
from _2_scribble2npy import scri2label
from _3_Labeling_for_superpixel import Label_superpixel
from _4_feature_from_RGB_and_location import feature_extracting
from _5_find_nearest import spatial_similarity
from _6_creating_graph import creating_graph
from _7_GCN_training import GCN_training, generate_mask

if __name__ == '__main__':
    # ---------------1--------------------
    Params = {
        'root': './image/',
        'tar_dir': './data/superpixel/',
        'tar_dir2': './data/superpixel_vis/'
    }
    if not os.path.exists(Params['tar_dir']):
        os.makedirs(Params['tar_dir'])
    if not os.path.exists(Params['tar_dir2']):
        os.makedirs(Params['tar_dir2'])

    # 定义超像素个数
    n_segments, compactness = 1800, 20
    paths = sorted(glob(Params['root'] + "*.jpg"))
    gen_superpixel_label(paths, n_segments, compactness,Params)

    # ---------------2--------------------
    Params = {
        'root': './scribble/',
        'tar_dir': './data/scribble_np/',
    }
    if not os.path.exists(Params['tar_dir']):
        os.makedirs(Params['tar_dir'])
    paths = sorted(glob(Params['root'] + "*.jpg"))
    scri2label(paths, Params)
    # ---------------3--------------------
    Params = {
        'scribble_np': './data/scribble_np/',
        'superpixel': './data/superpixel/',
        'sp_index2label': './data/sp_index2label/'
    }
    if not os.path.exists(Params['sp_index2label']):
        os.makedirs(Params['sp_index2label'])
    path1 = sorted(glob(Params['scribble_np'] + "*.npy"))
    path2 = sorted(glob(Params['superpixel'] + "*.npy"))
    Label_superpixel(path1, path2, Params)

    # ---------------4--------------------
    Params = {
        'image': './image/',  # 原始图像 用于提供RGB信息和空间位置信息
        'superpixel': './data/superpixel/',  # 超像素mask，用于标识某个超像素在原图中包含哪些像素
        'sp_index2label': './data/sp_index2label/',  # 用于提供有多少个超像素
        'features': "./data/features/"
    }
    if not os.path.exists(Params['features']):
        os.makedirs(Params['features'])
    path1 = sorted(glob(Params['image'] + "*.jpg"))
    path2 = sorted(glob(Params['superpixel'] + "*.npy"))
    path3 = sorted(glob(Params['sp_index2label'] + "*.npy"))
    feature_extracting(path1, path2, path3, Params)

    # ---------------5--------------------
    Params = {
        'features': './data/features/',  # 所有超像素的特征
        'connectivity': './data/connectivity/'  # 所有超像素之间的空间相似性
    }
    if not os.path.exists(Params['connectivity']):
        os.makedirs(Params['connectivity'])
    path = sorted(glob(Params['features'] + "*.npy"))
    spatial_similarity(path, Params)

    # ---------------6--------------------
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
    creating_graph(features_path, connectivity_path, sp_index2label_path, Params)

    # ---------------7--------------------
    Params = {
        'graph': './data/graph/',  # 图相关数据
        'superpixel': './data/superpixel/',
        'image': './image/',
        'scribble': './scribble/',
        'result': './result/'
    }
    if not os.path.exists(Params['result']):
        os.makedirs(Params['result'])
    super_paths = sorted(glob(Params['superpixel'] + "*.npy"))
    im_paths = sorted(glob(Params['image'] + "*.jpg"))
    sci_paths = sorted(glob(Params['scribble'] + "*.jpg"))
    model, adj, features, idx_test = GCN_training(Params, epochs=100, dropout=0.3)
    model.eval()
    gcn_th = 0.3
    output = model(features, adj)
    graph_prediction = (output > gcn_th).cpu().numpy().astype(np.uint8)
    generate_mask(graph_prediction, super_paths, im_paths, sci_paths, Params)
