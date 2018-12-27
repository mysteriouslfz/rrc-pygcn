#coding:utf-8
import numpy as np
import scipy.sparse as sp
import torch
import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(dataset="features-adj-labels"):
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))

    adj = np.load('/home/ubuntu/zhangzhihao/renrenche/usr_brows_v4/network/data_prepare_by_month/matrix.npy')
    # np.savetxt('adj_threshold.txt', adj.flatten(), fmt='%s')
    threshold = 3
    adj = adj_threshold(adj, threshold)  # 转化为binary
    adj = sp.csr_matrix(adj)   #sparse邻接矩阵
    # np.savetxt('adj_rrc.txt', adj.todense(), fmt='%s')

    features = pd.read_csv('/home/ubuntu/zhangzhihao/renrenche/usr_brows_v4/network/data_prepare_by_month/15_model_id_feature_transform_final.csv', header=None)
    features = np.array(features.values)
    # features = sp.csr_matrix(features, dtype=np.float32)   #sparse邻接矩阵
    # np.savetxt('featrues_rrc.txt', features.todense(), fmt='%s')

    labels = pd.read_csv('/home/ubuntu/zhangzhihao/renrenche/usr_brows_v4/network/data_prepare_by_month/15_model_id_label.csv')
    labels = labels['ave_price'].values
    labels = np.array(labels, dtype=np.float32).reshape((len(labels),1))

    # 标准化
    features = normalize_col(features)   #列标准化
    # features = normalize(features)   #行标准化
    adj = normalize(adj + sp.eye(adj.shape[0]))   #行标准化（邻接矩阵对角加1）

    # np.savetxt('featrues_normalize_rrc.txt', features.todense(), fmt='%s')
    # np.savetxt('adj_normalize_rrc.txt', adj.todense(), fmt='%s')

    X = range(labels.shape[0])
    idx_train, idx_test = train_test_split(X, test_size=0.2, random_state=0)#

    # features = torch.FloatTensor(np.array(features.todense()))
    features = torch.FloatTensor(np.array(features))
    labels = torch.FloatTensor(labels)   
    adj = sparse_mx_to_torch_sparse_tensor(adj)
    # adj = torch.FloatTensor(adj)


    idx_train = torch.LongTensor(idx_train)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_test

def accuracy(output, labels):
    preds = output.data.numpy()
    correct = labels.data.numpy()
    correct = np.abs(correct-preds).sum()
    return correct / len(labels)

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)   # .tocoo()转化为coo_matrix形式 .row .col .data
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))   # np.vatack（按照行顺序）的把数组给堆叠起来
    values = torch.from_numpy(sparse_mx.data)    # 将np.ndarray 转换为pytorch的 Tensor
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))   #加和列，得到每行的和的矩阵
    r_inv = np.power(rowsum, -1).flatten()   #取列的倒数并拉平
    r_inv[np.isinf(r_inv)] = 0.   #把倒数后为inf的行置0
    r_mat_inv = sp.diags(r_inv)   #将行加和并取倒数的列表形成对角矩阵
    mx = r_mat_inv.dot(mx)   #矩阵乘，最后相当于把当前矩阵的每个数字乘以该数字对应行和的倒数
    return mx

def normalize_col(mx):
    """Row-normalize sparse matrix"""
    colsum = np.array(mx.sum(0))   #每一列求和，得到每列的和的矩阵
    with np.errstate(divide='ignore'):
        r_inv = np.power(colsum, -1).flatten()   #取列的倒数并拉平
        r_inv[np.isinf(r_inv)] = 0.   #把倒数后为inf的行置0
    r_mat_inv = sp.diags(r_inv)   #将行加和并取倒数的列表形成对角矩阵
    mx =mx.dot(r_mat_inv)   #矩阵乘，最后相当于把当前矩阵的每个数字乘以该数字对应行和的倒数
    return mx

def normalize_collll(features):
    """Row-normalize sparse matrix"""
    my = features[:,0:48]
    mx = features[:,48:features.shape[1]]
    colsum = np.array(mx.sum(0))   #每一列求和，得到每列的和的矩阵
    with np.errstate(divide='ignore'):
        r_inv = np.power(colsum, -1).flatten()   #取列的倒数并拉平
        r_inv[np.isinf(r_inv)] = 0.   #把倒数后为inf的行置0
    r_mat_inv = sp.diags(r_inv)   #将行加和并取倒数的列表形成对角矩阵
    mx =mx.dot(r_mat_inv)   #矩阵乘，最后相当于把当前矩阵的每个数字乘以该数字对应行和的倒数

    feature = np.concatenate((my, mx.todense()), axis=1)
    return feature

def normalize_col(features):
    """Row-normalize sparse matrix"""
    my = features[:,0:48]
    mx = features[:,48:features.shape[1]]

    colsum = np.array(mx.sum(0))   #每一列求和，得到每列的和的矩阵
    with np.errstate(divide='ignore'):
        r_inv = np.power(colsum, -1).flatten()  #取列的倒数并拉平
    r_inv[np.isinf(r_inv)] = 0.
    r_inv = np.tile(r_inv, (mx.shape[0], 1))
    
    mx = np.multiply(mx, r_inv)
    features = np.concatenate((my, mx), axis=1)
    
    return features

def adj_threshold(adj, threshold):
    adj =np.where(adj < threshold, 0, adj)
    adj =np.where(adj >= threshold, 1, adj)
    
    return adj



if __name__ == '__main__':
    load_data(dataset="features-adj-labels")
