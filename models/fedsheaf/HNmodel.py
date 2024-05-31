# Copyright 2024 Wenfei Liang. 
# All rights reserved. 

import torch
import torch.nn as nn
import torch.nn.functional as F

class GNNHyperNetwork(nn.Module):
    def __init__(self, num_clients, feature_dim, hidden_dim, gcn_layer_dims, hn_dropout):
        """
        Args:
            feature_dim (int): Dimension of the input feature to the hypernetwork.
            hidden_dim (int): Dimension of the hidden layer in the hypernetwork.
            gcn_layer_dims (list of tuples): Dimensions of each layer in the GCN model. 
                                            Each tuple is (in_features, out_features).
        """
        super().__init__()

        # Store dimensions
        self.num_clients = num_clients

        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.gcn_layer_dims = gcn_layer_dims
        self.hn_dropout = hn_dropout

        # define attention layer
        self.attention_matrix = nn.Parameter(torch.eye(num_clients))
        self.dropout = nn.Dropout(p=self.hn_dropout)

        # Define MLP layers
        self.layer1 = nn.Linear(feature_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.relu = nn.ReLU(inplace=True)

        # Calculate total parameters for GCN layers
        total_params = sum(in_f * out_f + out_f for in_f, out_f in gcn_layer_dims)

        # Define the final layer to generate GCN parameters
        self.param_generator = nn.Linear(hidden_dim, total_params)

    def forward(self, features):
        weighted_embeddings = torch.matmul(self.attention_matrix, features)
        weighted_embeddings = self.dropout(weighted_embeddings)

        x = self.relu(self.layer1(weighted_embeddings))
        x = F.dropout(x, p=self.hn_dropout, training=self.training)
        x = self.relu(self.layer2(x))
        x = F.dropout(x, p=self.hn_dropout, training=self.training)

        # Generate parameters
        gcn_params = self.param_generator(x)

        return gcn_params