# Copyright 2025 Wenfei Liang. 
# All rights reserved. 

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GCN(nn.Module):
    def __init__(self, n_feat=10, n_dims=128, n_clss=10, args=None):
        super().__init__()
        self.n_feat = n_feat
        self.n_dims = n_dims
        self.n_clss = n_clss
        self.args = args

        self.conv1 = GCNConv(self.n_feat, self.n_dims, cached=False)
        self.conv2 = GCNConv(self.n_dims, self.n_dims, cached=False)
        self.clsif = nn.Linear(self.n_dims, self.n_clss)

    def forward(self, data, is_proxy=False):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.args.client_dropout, training=self.training)
        x = self.conv2(x, edge_index)
        if is_proxy == True: 
            return x
        x = F.relu(x)
        x = F.dropout(x, p=self.args.client_dropout, training=self.training)
        x = self.clsif(x)
        return x
