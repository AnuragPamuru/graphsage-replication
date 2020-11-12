#importing libraries
from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

#implement fully connected network
class Fully1Net(nn.Module):
    def __init__(self):
        super(Fully1Net, self).__init__()
        self.fc = nn.Linear(1433, 7, bias=False)
    def forward(self, x, adj_hat):
        return self.fc(x.view(x.size(0), -1))

#implement fully connected network with a hidden layer
class Fully2Net(nn.Module):
    def __init__(self):
        super(Fully2Net, self).__init__()
        num_hidden = 16
        self.fc1 = nn.Linear(1433, num_hidden, bias=False)
        self.fc2 = nn.Linear(num_hidden, 7, bias=False)
        
    def forward(self, x, adj_hat):
        out = self.fc1(x.view(x.size(0), -1))
        #ReLU activation function
        out = F.relu(out)
        out2 = self.fc2(out.view(out.size(0), -1))
        return out2
        
#implement graph convolutional network
class Graph1Net(nn.Module):
    def __init__(self):
        super(Graph1Net, self).__init__()
        self.fc = nn.Linear(1433, 7, bias=False)


    def forward(self, x, adj_hat):

        avg_neighbor_features = (torch.mm(adj_hat, x))
        
        return self.fc(avg_neighbor_features)

#implement graph convolutional network with a hidden layer
class Graph2Net(nn.Module):
    def __init__(self):
        super(Graph2Net, self).__init__()
        num_hidden = 16
        self.fc1 = nn.Linear(1433, num_hidden, bias=False)
        self.fc2 = nn.Linear(num_hidden, 7, bias=False)

    def forward(self, x, adj_hat):
        avg_neighbor_features = torch.mm(adj_hat, x)
        neighbor_out = self.fc1(avg_neighbor_features)
        #ReLU activation function
        neighbor_out = F.relu(neighbor_out)
        avg_neighbor_features2 = torch.mm(adj_hat, neighbor_out)
        neighbor_out = self.fc2(avg_neighbor_features2)
        return neighbor_out
