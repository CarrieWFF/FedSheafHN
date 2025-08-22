import torch
import random
import numpy as np

import metispy as metis

import torch_geometric
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_adj, dense_to_sparse

from utils import get_data, split_train, torch_save

data_path = '../../../datasets'
print('data_path', data_path)
ratio_train = 0.4
seed = 123
clients = [10,20]

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

def generate_data(dataset, n_clients):
    data = split_train(get_data(dataset, data_path), dataset, data_path, ratio_train, 'disjoint', n_clients)
    split_subgraphs(n_clients, data, dataset)

def split_subgraphs(n_clients, data, dataset):
    G = torch_geometric.utils.to_networkx(data)
    n_cuts, membership = metis.part_graph(G, n_clients)
    assert len(list(set(membership))) == n_clients
    
    for client_id in range(n_clients):
        client_indices = np.where(np.array(membership) == client_id)[0]
        edge_index_from = data.edge_index[0, :]
        edge_index_to = data.edge_index[1, :]
        mask = np.isin(edge_index_from.numpy(), client_indices) & np.isin(edge_index_to.numpy(), client_indices)
        client_edge_index = data.edge_index[:, mask]

        idx_mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(client_indices)}
        client_edge_index = torch.tensor([idx_mapping[idx.item()] for idx in client_edge_index.flatten()], dtype=torch.long)
        client_edge_index = client_edge_index.view(2, -1)

        client_x = data.x[client_indices]
        client_y = data.y[client_indices]
        client_train_mask = data.train_mask[client_indices]
        client_val_mask = data.val_mask[client_indices]
        client_test_mask = data.test_mask[client_indices]


        client_data = Data(
            x = client_x,
            y = client_y,
            edge_index=client_edge_index,
            train_mask = client_train_mask,
            val_mask = client_val_mask,
            test_mask = client_test_mask
        )
        assert torch.sum(client_train_mask).item() > 0

        torch_save(data_path, f'{dataset}_disjoint/{n_clients}/partition_{client_id}.pt', {
            'client_data': client_data,
            'client_id': client_id
        })
        

for n_clients in clients:
    # generate_data(dataset='Cora', n_clients=n_clients)
    # generate_data(dataset='CiteSeer', n_clients=n_clients)
    # generate_data(dataset='PubMed', n_clients=n_clients)
    # generate_data(dataset='Computers', n_clients=n_clients)
    # generate_data(dataset='Photo', n_clients=n_clients)
    generate_data(dataset='ogbn-arxiv', n_clients=n_clients)
