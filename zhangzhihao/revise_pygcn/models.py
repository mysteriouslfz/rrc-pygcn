#coding:utf-8

import torch.nn as nn
import torch.nn.functional as F
from layers import GraphConvolution

# linear 
class GCN(nn.Module):
    def __init__(self, nfeat, nhid, n_output, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, 64)
        self.predict = nn.Linear(64, n_output)   # output layer
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc2(x, adj))      # activation function for hidden layer
        x = self.predict(x)             # linear output
        return x

# 原始版本
# class GCN(nn.Module):
#     def __init__(self, nfeat, nhid, n_output, dropout):
#         super(GCN, self).__init__()

#         self.gc1 = GraphConvolution(nfeat, nhid)
#         self.gc2 = GraphConvolution(nhid, n_output)
#         self.dropout = dropout

#     def forward(self, x, adj):
#         x = F.relu(self.gc1(x, adj))
#         x = F.dropout(x, self.dropout, training=self.training)
#         x = self.gc2(x, adj)

#         return x

# 多个gcn层
# class GCN(nn.Module):
#     def __init__(self, nfeat, nhid, n_output, dropout):
#         super(GCN, self).__init__()

#         self.gc1 = GraphConvolution(nfeat, nhid)
#         self.gc2 = GraphConvolution(nhid, 64)
#         self.gc3 = GraphConvolution(64, n_output)
#         self.dropout = dropout

#     def forward(self, x, adj):
#         x = F.relu(self.gc1(x, adj))
#         x = F.dropout(x, self.dropout, training=self.training)
#         x = self.gc2(x, adj)
#         x = self.gc3(x, adj)

#         return x

# 多个gcn层+linear
# class GCN(nn.Module):
#     def __init__(self, nfeat, nhid, n_output, dropout):
#         super(GCN, self).__init__()

#         self.gc1 = GraphConvolution(nfeat, nhid)
#         self.gc2 = GraphConvolution(nhid, 64)
#         self.gc3 = GraphConvolution(64, 64)
#         self.dropout = dropout
#         self.predict = nn.Linear(64, n_output) 

#     def forward(self, x, adj):
#         x = F.relu(self.gc1(x, adj))
#         x = F.dropout(x, self.dropout, training=self.training)
#         x = self.gc2(x, adj)
#         x = F.relu(self.gc3(x, adj))      # activation function for hidden layer
#         x = self.predict(x)             # linear output

#         return x

# class GCN(nn.Module):
#     def __init__(self, nfeat, nhid, n_output, dropout):
#         super(GCN, self).__init__()

#         self.gc1 = GraphConvolution(nfeat, 64)
#         # self.gc2 = GraphConvolution(nhid, 64)
#         self.predict = nn.Linear(64, n_output)   # output layer
#         self.dropout = dropout

#     def forward(self, x, adj):
#         # x = F.relu(self.gc1(x, adj))
#         # x = F.dropout(x, self.dropout, training=self.training)
#         x = F.relu(self.gc1(x, adj))      # activation function for hidden layer
#         x = self.predict(x)             # linear output
#         return x
