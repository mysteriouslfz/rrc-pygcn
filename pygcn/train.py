from __future__ import division
from __future__ import print_function

import time
import argparse
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import MSELoss

from utils import load_data, load_kdd_data, accuracy, accuracy_new
from models import GCN,GCN_R

MODE = 2

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=200,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=16,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Load data
if MODE == 1:
    adj, features, labels, idx_train, idx_val, idx_test = load_data()
elif MODE == 2:
    adj, features, labels, idx_train, idx_val, idx_test = load_kdd_data()

# Model and optimizer
if MODE == 1:
    model = GCN(nfeat=features.shape[1],
                nhid=args.hidden,
                nclass=labels.max().item() + 1,
                dropout=args.dropout)
elif MODE == 2:
    model = GCN_R(nfeat=features.shape[1],
                nhid=args.hidden,
                dropout=args.dropout)

optimizer = optim.SGD(model.parameters(),
                       lr=args.lr, weight_decay=args.weight_decay)

if args.cuda:   #GPU支持
    model.cuda()
    features = features.cuda()
    adj = adj.cuda()
    labels = labels.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()


def train(epoch,mode):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    output = model(features, adj)
    if mode == 1:
        loss_train = F.nll_loss(output[idx_train], labels[idx_train])
        acc_train = accuracy(output[idx_train], labels[idx_train])
    elif mode == 2:
        loss_func = MSELoss()
        loss_train = loss_func(output[idx_train], labels[idx_train])
        acc_train = accuracy_new(output[idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()

    if not args.fastmode:
        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
        model.eval()
        output = model(features, adj)

    if mode == 1:
        loss_val = F.nll_loss(output[idx_val], labels[idx_val])
        acc_val = accuracy(output[idx_val], labels[idx_val])
    elif mode == 2:
        loss_func = MSELoss()
        loss_val = loss_func(output[idx_val], labels[idx_val])
        acc_val = accuracy_new(output[idx_val], labels[idx_val])
    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'acc_train: {:.4f}'.format(acc_train.item()),
          'loss_val: {:.4f}'.format(loss_val.item()),
          'acc_val: {:.4f}'.format(acc_val.item()),
          'time: {:.4f}s'.format(time.time() - t))


def test(mode):
    model.eval()
    output = model(features, adj)
    if mode == 1:
        loss_test = F.nll_loss(output[idx_test], labels[idx_test])
        acc_test = accuracy(output[idx_test], labels[idx_test])
    elif mode == 2:
        loss_func = MSELoss()
        loss_test = loss_func(output[idx_test], labels[idx_test])
        acc_test = accuracy_new(output[idx_test], labels[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))


# Train model
t_total = time.time()
for epoch in range(args.epochs):
    train(epoch,MODE)
print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

# Testing
test(MODE)
