import math

import torch

from torch.nn.parameter import Parameter
import torch.nn as nn
import torch.nn.functional as F

class GraphConvolution(nn.Module):
    """
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
        # Normalization as per (Kipf & Welling, ICLR 2017)
        D = adj.sum(1)  # nodes degree (N,)
        D_hat = (D + 1e-5) ** (-0.5)
        adj_hat = D_hat.view(-1, 1) * adj * D_hat.view(1, -1)  # N,N

        # Some additional trick I found to be useful
        adj_hat[adj_hat > 0.0001] = adj_hat[adj_hat > 0.0001] - 0.2

        #A * X * W
        output = torch.mm(adj_hat, input)
        output = torch.mm(output, self.weight)

        if self.bias is not None:
            return output + self.bias
        else:
            return output

class GCNLPAConv(nn.Module):
    """
    A GCN-LPA layer. Please refer to: https://arxiv.org/abs/2002.06755
    """

    def __init__(self, in_features, out_features, adj, bias=True):
        super(GCNLPAConv, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        self.adjacency_mask = Parameter(adj.clone())

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x, adj, y):
        y = y.view(len(y), 1).float()
        # W * x
        support = torch.mm(x, self.weight)
        # Hadamard Product: A' = Hadamard(A, M)
        adj = adj * self.adjacency_mask
        # Row-Normalize: D^-1 * (A')
        adj = F.normalize(adj, p=1, dim=1)

        # output = D^-1 * A' * X * W
        output = torch.mm(adj, support)
        # y' = D^-1 * A' * y
        print("D^-1 * A':", adj.size())
        print("y:", y.size())
        y_hat = torch.mm(adj, y)

        if self.bias is not None:
            return output + self.bias, y_hat
        else:
            return output, y_hat
