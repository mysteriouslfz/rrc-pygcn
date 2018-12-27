#coding:utf-8
# from __future__ import division
# from __future__ import print_function

import time
import argparse
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim

from utils import load_data, accuracy
from models import GCN

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=1000,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=64,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()   # torch.cuda.is_available() == False

np.random.seed(args.seed)
torch.manual_seed(args.seed)   # 为CPU设置种子用于生成随机数，以使得结果是确定的
if args.cuda:
    torch.cuda.manual_seed(args.seed)  # 为当前GPU设置随机种子；如果使用多个GPU,torch.cuda.manual_seed_all()为所有的GPU设置种子

# Load data
adj, features, labels, idx_train, idx_test = load_data()
print adj.shape,features.shape,labels.shape,idx_test.shape
# print adj.shape, adj  # adj:torch.sparse_coo
# print features.shape, features # features:dense
# print labels.shape, labels  # label:index


# Model and optimizer
model = GCN(nfeat=features.shape[1],
            nhid=args.hidden,
            n_output=1,
            dropout=args.dropout)
optimizer = optim.Adam(model.parameters(),
                       lr=args.lr, weight_decay=args.weight_decay)
loss_func = torch.nn.MSELoss() 

'''
print model
GCN(
  (gc1): GraphConvolution (1433 -> 16)
  (gc2): GraphConvolution (16 -> 7)
)
'''

if args.cuda:
    model.cuda()
    features = features.cuda()
    adj = adj.cuda()
    labels = labels.cuda()
    idx_train = idx_train.cuda()
    idx_test = idx_test.cuda()


def train(epoch):
    t = time.time()
    model.train()            # 训练模式，对dropout和batch normalization的操作在训练和测试的时候是不一样的
    
    # forward
    output = model(features, adj)
    loss_train = loss_func(output[idx_train], labels[idx_train])   # loss function
    acc_train = accuracy(output[idx_train], labels[idx_train])
    loss_test = loss_func(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])

    # backward
    optimizer.zero_grad()    # clear gradients for next train
    loss_train.backward()    # backpropagation, compute gradients
    optimizer.step()         # apply gradients

    # print 'loss_train:',type(loss_train.data.numpy())
    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(loss_train.data.numpy()),
          'mae_train: {:.4f}'.format(acc_train.item()),
          'loss_test: {:.4f}'.format(loss_test.data.numpy()),
          'mae_test: {:.4f}'.format(acc_test.item()),
          'time: {:.4f}s'.format(time.time() - t))


def test():
    model.eval()  # 测试模式
    output = model(features, adj)
    loss_test = loss_func(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.data.numpy()),
          "mae= {:.4f}".format(acc_test.item()))


# Train model
t_total = time.time()
for epoch in range(args.epochs):
    train(epoch)
print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

# Testing
test()
