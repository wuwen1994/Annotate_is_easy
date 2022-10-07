import torch
import numpy as np
from GCN import GCN
import scipy.sparse as sp
import torch.optim as optim
from glob import glob
import cv2 as cv
import os
from utils import supervised_loss, normalize, normalize_adj, sparse_mx_to_torch_sparse_tensor, accuracy


def load_data(Params):
    val_portion = 0.20

    graph_path = Params['graph'] + "graph.npy"
    weights_path = Params['graph'] + "graph_weights.npy"
    features_path = Params['graph'] + "graph_node_features.npy"
    labels_path = Params['graph'] + "graph_ground_truth.npy"
    mask_path = Params['graph'] + "unc_mask.npy"  # 不参与训练但要参与测试

    graph = np.load(graph_path)
    weights = np.load(weights_path)
    features = np.load(features_path)

    test_mask = np.load(mask_path)  # 不参与训练但要参与测试
    full_mask = 1 - test_mask  # 参与训练的样本

    labels = np.load(labels_path)  # ROI区域所有节点的CNN初始预测值, 其中可靠样本拿走标签，不可靠样本标签依赖后续的传播
    num_nodes = labels.shape[0]  # 节点的个数
    adj = sp.coo_matrix((weights, (graph[:, 0], graph[:, 1])), shape=(num_nodes, num_nodes))  # 新建一个矩阵，将权重放到相应的位置
    features = sp.coo_matrix(features)  # numpy.ndarray -> tuple
    working_nodes = np.where(full_mask != 0)[0]  # 可靠的节点 (12594,)
    random_arr = np.random.uniform(low=0, high=1, size=working_nodes.shape)  # 生成随机数组(12594,)

    features = normalize(features)  # (13359, 5) 特征归一化    特征矩阵
    adj = normalize_adj(adj + sp.eye(adj.shape[0]))  # 权重矩阵  加上 对角 矩阵?

    # 确定训练集和验证集节点的ID
    idx_train = working_nodes[random_arr > val_portion]
    idx_val = working_nodes[random_arr <= val_portion]
    # 测试集
    idx_test = np.where(test_mask != 0)

    # 转化数据类型为Tensor, 准备训练
    features = torch.FloatTensor(np.array(features.todense()))  # sparse.matrix -> Tensor  [13359, 5]
    labels = torch.FloatTensor(labels)  # ndarray -> Tensor  [13359]

    adj = sparse_mx_to_torch_sparse_tensor(adj)  # sparse.matrix -> Tensor [13359, 13359]

    idx_train = torch.LongTensor(idx_train)  # ndarray ->  Tensor   [10047]
    idx_val = torch.LongTensor(idx_val)  # ndarray ->  Tensor  [2547]
    idx_test = torch.LongTensor(idx_test[0])  # tuple -> Tensor  [765]

    return adj, features, labels, idx_train, idx_val, idx_test


def train(model, optimizer, epoch, adj, features, labels, idx_train, idx_val, idx_test):
    # 训练
    model.train()
    optimizer.zero_grad()
    output = model(features, adj)
    loss_train = supervised_loss(output[idx_train], labels[idx_train])
    acc_train = accuracy(output[idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()

    # 测试
    model.eval()
    output = model(features, adj)
    # loss_val = F.nll_loss(output[idx_val], labels[idx_val])
    loss_val = supervised_loss(output[idx_val], labels[idx_val])
    acc_val = accuracy(output[idx_val], labels[idx_val])
    print('Epoch: {:04d}'.format(epoch + 1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'acc_train: {:.4f}'.format(acc_train.item()),
          'loss_val: {:.4f}'.format(loss_val.item()),
          'acc_val: {:.4f}'.format(acc_val.item()))


def GCN_training(Params, epochs, dropout):
    # 训练设置
    seed = 42
    lr = 1e-2
    weight_decay = 1e-5
    hidden = 32
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)

    # 加载数据
    adj, features, labels, idx_train, idx_val, idx_test = load_data(Params)
    # adj:   权重矩阵+对角矩阵
    # features: 特征矩阵
    # labels: CNN初始预测->作为GCN中的label信息的提供者
    # y_test: real GT
    # idx_train, idx_val, idx_test: 训练集，验证集，和测试集

    # 加载模型

    model = GCN(nfeat=features.shape[1],  # 输入5维度
                nhid=hidden,  # 隐藏层32维度
                nclass=1,  # 输出为1维
                dropout=dropout)  # dropout 0.5

    optimizer = optim.Adam(model.parameters(),
                           lr=lr, weight_decay=weight_decay)

    # 导入cuda
    model.cuda()
    features = features.cuda()
    adj = adj.cuda()
    labels = labels.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()

    # 模型训练
    torch.set_grad_enabled(True)
    model.eval()
    for epoch in range(epochs):
        train(model, optimizer, epoch, adj, features, labels, idx_train, idx_val, idx_test)
    return model, adj, features, idx_test


def generate_mask(graph_prediction, super_paths, im_paths, sci_paths, Params):
    for i, p in enumerate(super_paths):
        image_path = im_paths[i]
        image = cv.imread(image_path,-1)
        sci_path = sci_paths[i]
        sci = cv.imread(sci_path)
        filename = p.split("\\")[-1].split(".")[0]
        superpixel = np.load(p)
        height, width = superpixel.shape[:]
        result = np.zeros((height, width), dtype=np.uint8)
        for index, prediction in enumerate(graph_prediction):
            h = np.where(superpixel == index)[0]
            w = np.where(superpixel == index)[1]
            result[h, w] = prediction * 255
        cv.imwrite(Params['result'] + filename + '.png', result)
        cv.imshow("shadow_image", image)
        cv.imshow("scibble annotations", sci)
        cv.imshow("generated shadow mask", result)
        cv.waitKey()


if __name__ == '__main__':
    Params = {
        'graph': './data/graph/',  # 图相关数据
        'superpixel': './data/superpixel/',
        'result': './data/result/'
    }
    if not os.path.exists(Params['result']):
        os.makedirs(Params['result'])
    path = sorted(glob(Params['superpixel'] + "*.npy"))
    model, adj, features, idx_test = GCN_training(epochs=700, dropout=0.3)
    model.eval()
    gcn_th = 0.3
    output = model(features, adj)
    graph_prediction = (output > gcn_th).cpu().numpy().astype(np.uint8)
    generate_mask(graph_prediction, path)
