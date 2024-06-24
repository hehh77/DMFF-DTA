import numpy as np
import rdkit
import rdkit.Chem as Chem
import networkx as nx
from build_vocab import WordVocab
import pandas as pd

import os
import torch.nn as nn

from utils import *

from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_add_pool
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from torch.nn import Sequential, Linear, ReLU
#############################


class SpatialGroupEnhance_for_1D(nn.Module):
    def __init__(self, groups = 32):
        super(SpatialGroupEnhance_for_1D, self).__init__()
        self.groups   = groups
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.weight   = Parameter(torch.zeros(1, groups, 1))
        self.bias     = Parameter(torch.ones(1, groups, 1))
        self.sig      = nn.Sigmoid()
    
    def forward(self, x): # (b, c, h)
        b, c, h = x.size()
        x = x.view(b * self.groups, -1, h)
        xn = x * self.avg_pool(x)
        xn = xn.sum(dim=1, keepdim=True)
        t = xn.view(b * self.groups, -1)
        t = t - t.mean(dim=1, keepdim=True)
        std = t.std(dim=1, keepdim=True) + 1e-5
        t = t / std
        t = t.view(b, self.groups, h)
        t = t * self.weight + self.bias
        t = t.view(b * self.groups, 1, h)
        x = x * self.sig(t)
        x = x.view(b, c, h)
        return x

class LinkAttention(nn.Module):
    def __init__(self, input_dim, n_heads):
        super(LinkAttention, self).__init__()
        self.query = nn.Linear(input_dim, n_heads)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, masks):
        query = self.query(x).transpose(1, 2) # (B,heads,seq_len)
        value = x # (B,seq_len,hidden_dim)

        minus_inf = -9e15 * torch.ones_like(query) # (B,heads,seq_len)
        e = torch.where(masks > 0.5, query, minus_inf)  # (B,heads,seq_len)
        a = self.softmax(e) # (B,heads,seq_len)

        out = torch.matmul(a, value) # (B,heads,seq_len) * (B,seq_len,hidden_dim) = (B,heads,hidden_dim)
        out = torch.mean(out, dim=1).squeeze() # (B,hidden_dim)
        return out, a
    
class GINConvNet(torch.nn.Module):
    def __init__(self, n_output=1, num_features_xd=128, num_features_xt=25,
                 n_filters=32, embed_dim=128, output_dim=128, dropout=0.2):
        super(GINConvNet, self).__init__()

        dim = 128
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.n_output = n_output
        # convolution layers
        nn1 = Sequential(Linear(num_features_xd, dim), ReLU(), Linear(dim, dim))
        self.conv1 = GINConv(nn1)
        self.bn1 = torch.nn.BatchNorm1d(dim)

        nn2 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv2 = GINConv(nn2)
        self.bn2 = torch.nn.BatchNorm1d(dim)

        nn3 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv3 = GINConv(nn3)
        self.bn3 = torch.nn.BatchNorm1d(dim)

        nn4 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv4 = GINConv(nn4)
        self.bn4 = torch.nn.BatchNorm1d(dim)

        nn5 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv5 = GINConv(nn5)
        self.bn5 = torch.nn.BatchNorm1d(dim)

        self.fc1_xd = Linear(dim, output_dim)

        # combined layers
        self.fc1 = nn.Linear(128, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.out = nn.Linear(256, self.n_output)  # n_output = 1 for regression task

    def forward(self, data):
        x, edge_index, batch = data.x.to(device), data.edge_index.to(device), data.batch.to(device)
        # target = data.target
        x = F.relu(self.conv1(x, edge_index)) # (B,seq_len,hidden_dim)
        x = self.bn1(x)
        x = F.relu(self.conv2(x, edge_index)) # (B,seq_len,hidden_dim)
        x = self.bn2(x)
        x = F.relu(self.conv3(x, edge_index)) # (B,seq_len,hidden_dim)
        x = self.bn3(x)
        x = F.relu(self.conv4(x, edge_index)) # (B,seq_len,hidden_dim)
        x = self.bn4(x)
        x = F.relu(self.conv5(x, edge_index)) # (B,seq_len,hidden_dim)
        x = self.bn5(x)
        x = global_add_pool(x, batch) # (B,hidden_dim)
        x = F.relu(self.fc1_xd(x)) # (B,hidden_dim)
        x = F.dropout(x, p=0.2, training=self.training) # (B,hidden_dim)
        # concat
        xc = x
        xc = self.fc1(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        xc = self.fc2(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        out = self.out(xc)
        return out


    
