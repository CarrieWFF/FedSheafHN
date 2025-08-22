# Copyright 2025 Wenfei Liang. 
# All rights reserved. 

import torch
import torch.nn as nn
import time

from torch_geometric.data import Data
from misc.utils import *
from modules.logger import Logger
from models.neural_sheaf.server.disc_models import DiscreteDiagSheafDiffusion
from models.fedsheaf.HNmodel import GNNHyperNetwork

class Server:
    def __init__(self, args, sd, gpu_server):
        self.args = args
        self.sd = sd
        self.gpu_id = gpu_server
        self.logger = Logger(self.args, self.gpu_id, is_server=True)
        
        self.model = DiscreteDiagSheafDiffusion
        self.model_hn = GNNHyperNetwork
        self.client_graph = {}
        self.log = {'total_val_acc': [], 'total_test_acc': []}

    def on_round_begin(self, curr_rnd):
        self.round_begin = time.time()
        self.curr_rnd = curr_rnd

    def round_end(self, curr_rnd, allclients, selected):
        val_acc_tmp = []
        test_acc_tmp = []

        for c_id in allclients:
            if c_id in selected: 
                val_acc_tmp.append(self.sd[c_id]['val_acc'])
                test_acc_tmp.append(self.sd[c_id]['test_acc'])
            del self.sd[c_id]

        self.log['total_val_acc'].append(sum(val_acc_tmp)/len(val_acc_tmp))
        self.log['total_test_acc'].append(sum(test_acc_tmp)/len(test_acc_tmp))
        print(f'in round test acc {self.log['total_test_acc'][-1]}')

    def procs_end(self):
        best_val = max(self.log['total_val_acc'])
        best_test = max(self.log['total_test_acc'])
        return best_val, best_test
        
        ################################ construct collaboration graph ##################
    def construct_graph(self, updated, curr_rnd):
        # Generate the collaboration graph with client graph-level embeddings
        client_embeddings = []

        for c_id in sorted(updated):
            tensor_embedding = self.sd[c_id]['functional_embedding']
            client_embeddings.append(tensor_embedding)
            del tensor_embedding
            del self.sd[c_id]['functional_embedding']

        embeddings = torch.cat(client_embeddings, dim=0)
        edges = [[i, j] for i in range(len(updated)) for j in range(len(updated)) if i != j]
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        self.client_graph = Data(x=embeddings, edge_index=edge_index).cuda(self.gpu_id)
        del client_embeddings
        del embeddings

        self.args.graph_size = self.client_graph.x.size(0)
        self.args.input_dim = self.client_graph.num_features
        self.args.output_dim = self.args.input_dim

        if self.args.server_sheaf_decay is None:
            self.args.server_sheaf_decay = self.args.server_weight_decay

    ###################################### train server model ########################################
    def train_server_GNN(self, curr_rnd):
        if curr_rnd == 0:
            self.model = self.model(self.client_graph.edge_index, self.args).cuda(self.gpu_id)
            sheaf_learner_params, other_params = self.model.grouped_parameters()
            self.optimizer_gnn = torch.optim.Adam([
                {'params': sheaf_learner_params, 'weight_decay': self.args.server_sheaf_decay},
                {'params': other_params, 'weight_decay': self.args.server_weight_decay}
            ], lr=self.args.server_lr)

            ###################### optimizer for hn ###########################################
            self.model_hn = self.model_hn(self.args.n_clients, self.client_graph.num_features, self.args.HN_hidden_dim, self.args.gcn_layer_dims, self.args.hn_dropout).cuda(self.gpu_id)
            self.optimizer_hn = torch.optim.Adam(self.model_hn.parameters(), lr=self.args.server_hn_lr)

        if hasattr(self, 'updated_embedding'):
            del self.updated_embedding
        self.optimizer_gnn.zero_grad()

        self.model.train()
        self.updated_embedding = self.model(self.client_graph.x)
        self.grad_tensor = torch.zeros_like(self.updated_embedding)

    def train_server_HN(self, curr_rnd):
        ############################ Hypernetwork generate model params ##################################
        self.optimizer_hn.zero_grad()
        self.model_hn.train()

        if hasattr(self, 'eb_tmp'):
            del self.eb_tmp

        self.eb_tmp = self.updated_embedding.clone()
        self.eb_tmp.requires_grad_(True)
        self.eb_tmp.retain_grad()
        self.gcn_params = self.model_hn(self.eb_tmp)
        
        for c_id in range(self.gcn_params.shape[0]):
            client_params_tmp = self.gcn_params[c_id, :]
            weights = {}
            pointer = 0
            for i, (in_f, out_f) in enumerate(self.args.gcn_layer_dims):
                weight_size = in_f * out_f
                bias_size = out_f
                weights[f'gcn{i+1}.weight'] = client_params_tmp[pointer:pointer + weight_size].view(in_f, out_f)
                pointer += weight_size
                weights[f'gcn{i+1}.bias'] = client_params_tmp[pointer:pointer + bias_size].view(out_f)
                pointer += bias_size

            self.sd[c_id] = {'generated model params': {k: v.clone().detach() for k, v in weights.items()}}

        ################################# update server model ############################################
    def update_server_HN(self, updated):
        collected_delta_params = []
        keys_order = ['gcn1.weight', 'gcn1.bias', 'gcn2.weight', 'gcn2.bias']

        for c_id in sorted(updated):
            delta_param = self.sd[c_id]['delta param']
            del self.sd[c_id]['delta param']

            flattened_params = []
            for key in keys_order:
                if key not in delta_param:
                    raise KeyError(f"Key '{key}' not found in delta_param")
                flattened_params.append(delta_param[key].view(-1))

            delta_gcn_params = torch.cat(flattened_params)
            delta_gcn_params = delta_gcn_params.view_as(self.gcn_params[c_id, :])
            collected_delta_params.append(delta_gcn_params)

        all_delta_params = torch.stack(collected_delta_params)

        self.optimizer_hn.zero_grad()
        gnet_grads = torch.autograd.grad(
            self.gcn_params, self.eb_tmp, grad_outputs=all_delta_params, retain_graph=True)

        self.grad_tensor = gnet_grads[0].clone()
        
        self.optimizer_hn.zero_grad()
        average_grads = torch.autograd.grad(
            self.gcn_params, self.model_hn.parameters(), grad_outputs=all_delta_params)

        for p, g in zip(self.model_hn.parameters(), average_grads):
            if p.grad is not None:
                p.grad.zero_()
            if g is not None:
                p.grad = g
        
        self.optimizer_hn.step()
        collected_delta_params.clear()

        self.optimizer_gnn.zero_grad()
        torch.autograd.backward(self.updated_embedding, grad_tensors=[self.grad_tensor])
        self.optimizer_gnn.step()
        self.save_state()

    def load_state(self):
        loaded = torch_load(self.args.checkpt_path, 'server_state.pt')
        set_state_dict(self.model, loaded['model'], self.gpu_id)
        set_state_dict(self.model_hn, loaded['model_hn'], self.gpu_id)
        self.optimizer_gnn.load_state_dict(loaded['optimizer_gnn'])
        self.optimizer_hn.load_state_dict(loaded['optimizer_hn'])
        self.log = loaded['log']

    def save_state(self):
        torch_save(self.args.checkpt_path, 'server_state.pt', {
            'model': get_state_dict(self.model),
            'model_hn': get_state_dict(self.model_hn),
            'optimizer_gnn': self.optimizer_gnn.state_dict(),
            'optimizer_hn': self.optimizer_hn.state_dict(),
            'log': self.log
        })
