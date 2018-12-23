import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot


def load_data(path="../data/cora/", dataset="cora"):
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))

    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),
                                        dtype=np.dtype(str))
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)   #得到features的稀疏矩阵
    labels = encode_onehot(idx_features_labels[:, -1])   #得到labels的one-hot矩阵

    # build graph
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)   #得到所有sample的编号
    idx_map = {j: i for i, j in enumerate(idx)}   #得到所有sample的编号与node对应的字典
    edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),
                                    dtype=np.int32)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape)   #得到边的列表
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)   #得到邻接稀疏矩阵（非对称）

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)   #得到对称邻接稀疏矩阵

    features = normalize(features)   #行标准化
    adj = normalize(adj + sp.eye(adj.shape[0]))   #行标准化（邻接矩阵对角加1）

    idx_train = range(140)
    idx_val = range(200, 500)
    idx_test = range(500, 1500)

    features = torch.FloatTensor(np.array(features.todense()))   #变为torch的稠密矩阵
    labels = torch.LongTensor(np.where(labels)[1])   #把one-hot的labels变回数字形式的label
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test

def load_kdd_data():
    features = pd.read_csv('../kdddata/15_model_id_feature_transform_final.csv',header=None)
    features = np.array(features.values)
    features = sp.csr_matrix(features, dtype=np.float32)   #得到features的稀疏矩阵

    labels = pd.read_csv('../kdddata/15_model_id_label.csv')
    labels = labels['ave_price'].values
    labels = np.array(labels).reshape((len(labels),1))   #得到price的label值

    adj = np.load('../kdddata/matrix.npy')
    adj = sp.csr_matrix(adj)
    # adj = adj + sp.eye(adj.shape[0])   #给对角线加上数字（或许原来的矩阵不要去掉selfloop更好？）

    features = normalize(features)   #行标准化
    adj = normalize(adj + sp.eye(adj.shape[0]))   #行标准化（邻接矩阵对角加1）

    idx_train = range(1200)
    idx_val = range(1200, 1600)
    idx_test = range(1600, adj.shape[0])

    features = torch.FloatTensor(np.array(features.todense()))   #变为torch的稠密矩阵
    labels = torch.FloatTensor(labels)   #变为torch的形式
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test

def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))   #加和列，得到每行的和的矩阵
    r_inv = np.power(rowsum, -1).flatten()   #取列的倒数并拉平
    r_inv[np.isinf(r_inv)] = 0.   #把倒数后为inf的行置0
    r_mat_inv = sp.diags(r_inv)   #将行加和并取倒数的列表形成对角矩阵
    mx = r_mat_inv.dot(mx)   #矩阵乘，最后相当于把当前矩阵的每个数字乘以该数字对应行和的倒数
    return mx


def accuracy(output, labels):   #先将输出转换为labels，再比较预测正确的数量
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

def accuracy_new(output, labels):
    preds = output
    correct_loss = abs(preds - labels)
    correct = correct_loss.sum()
    return correct / len(labels)

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)   #把计算后的稀疏矩阵转为高效coo矩阵
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))   #把稀疏矩阵的行和列全部拿出来并竖排堆叠起来
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)   #把普通的稀疏矩阵转换为Torch的稀疏矩阵