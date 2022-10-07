import json
import numpy as np
import torch
import scipy.sparse as sp
import torch.nn.functional as F

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)

def save_json(save_path, file):
    with open(save_path, 'w') as f:
        json.dump(file, f, cls=NpEncoder)


def supervised_loss(prediction, label):
    # mask = label.clone()
    # num_positive = torch.sum((mask == 1).float()).float()
    # num_negative = torch.sum((mask == 0).float()).float()
    #
    # mask[mask == 1] = num_negative / (num_positive + num_negative)
    # mask[mask == 0] = num_positive / (num_positive + num_negative)
    cost = F.binary_cross_entropy(prediction, label)
    return cost  # torch.cuda.FloatTensor

def normalize(mx):
    """行归一化"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def normalize_adj(adj):
    """行归一化"""
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def accuracy(output, labels):
    preds = (output > 0.5).type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)
