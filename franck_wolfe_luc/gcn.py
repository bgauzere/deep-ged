import math
import torch
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module


class GraphConvolution(Module):
    """
    From Thomas Kipf repo
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + str(self.in_features) + ' -> ' \
            + str(self.out_features) + ')'


class GraphPooling(Module):
    """
    Simple Graph pooling layer
    """

    # def __init__(self):  # , out_features):
    #     super(GraphPooling, self).__init__()
    #     # self.out_features = out_features

    def forward(self, x):
        output = torch.sum(x, dim=0)
        #output = torch.mean (x, dim=0)

        return output

    def __repr__(self):
        return self.__class__.__name__

import torch.nn as nn
import torch.nn.functional as F
from layers import GraphConvolution, GraphPooling


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, 2*nhid)
        self.dropout = dropout
        self.pooling = GraphPooling()
        self.fc3 = nn.Linear(2*nhid, nhid)
        self.fc4 = nn.Linear(nhid, nclass)

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)

        x = F.relu(self.gc2(x, adj))
        print(x)
        x = F.relu(self.pooling(x))
        x = F.relu(self.fc3(x))
        x = F.log_softmax(self.fc4(x), dim=0)
        return x

class GCN2(nn.Module):
    def __init__(self, nfeat, final_dim, dropout):
        super(GCN2, self).__init__()
        nhid=int(final_dim/2)
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, final_dim)
        self.dropout = dropout
        
    def forward(self, x, adj):
        
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)

        x = F.relu(self.gc2(x, adj))
        
        return x
    