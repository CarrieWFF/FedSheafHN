# Copyright 2024 Wenfei Liang. 
# All rights reserved. 

import torch
import torch.nn as nn
import numpy as np
from collections import OrderedDict

import time
import torch.nn.functional as F
from misc.utils import *
from data.loader import DataLoader
from modules.logger import Logger
from models.fedsheaf.GCNmodel import GCN
    
class Client:
    def __init__(self, args, g_id, sd, client_id):
        self.sd = sd
        self.gpu_id = g_id
        self.args = args 
        self._args = vars(self.args) 
        self.client_id = client_id
        self.loader = DataLoader(self.args)
        self.logger = Logger(self.args, self.gpu_id)

        self.model = GCN
        
        if args.dataset == 'Cora':
            self.args.n_classes = 7 
        elif args.dataset == 'CiteSeer': 
            self.args.n_classes = 6
        elif args.dataset == 'PubMed':
            self.args.n_classes = 3
        elif args.dataset == 'ogbn-arxiv':
            self.args.n_classes = 40
        elif args.dataset == 'Computers':
            self.args.n_classes = 10
        elif args.dataset == 'Photo':
            self.args.n_classes = 8
        
    def init_state(self):
        for _, batch in enumerate(self.loader.pa_loader):
            self.args.graph_size = batch.x.size(0)
            self.args.input_dim = batch.num_features
            
            self.model = self.model(self.args.input_dim, self.args.client_hidden_dim, self.args.n_classes, self.args).cuda(self.gpu_id)
            self.parameters = list(self.model.parameters())

        self.optimizer = torch.optim.Adam(self.parameters, lr=self.args.client_lr, weight_decay=self.args.client_weight_decay)

    def save_state(self):
        torch_save(self.args.checkpt_path, f'{self.client_id}_state.pt', {
            'optimizer': self.optimizer.state_dict(),
            'model': get_state_dict(self.model)
        })

    def load_state(self):
        loaded = torch_load(self.args.checkpt_path, f'{self.client_id}_state.pt')
        set_state_dict(self.model, loaded['model'], self.gpu_id)
        self.optimizer.load_state_dict(loaded['optimizer'])

    def on_receive_message(self, curr_rnd, message_type):
        self.curr_rnd = curr_rnd
        if message_type == 'client_generate_vector_start': 
            print(f'[client{self.client_id}] round{self.curr_rnd} generate vector start')
        if message_type == 'client_train_on_generated_model_prams':
            print(f'[client{self.client_id}] round{self.curr_rnd} begin training on generated model params')

    def generate_vector(self, client_id):
        self.model.train()

        for _, batch in enumerate(self.loader.pa_loader):
            batch = batch.cuda(self.gpu_id)
            for epoch in range(self.args.client_vector_epochs):
                self.optimizer.zero_grad()
                out = self.model(batch)
                train_lss = F.cross_entropy(out[batch.train_mask], batch.y[batch.train_mask])
                train_lss.backward()
                self.optimizer.step()

                if epoch == self.args.client_vector_epochs - 1: 
                    self.model.eval()
                    with torch.no_grad():
                        embedding_tmp = self.model(batch, is_proxy=True)
        
        average_embedding = torch.mean(embedding_tmp, dim=0, keepdim=True)

        self.sd[client_id] = {'functional_embedding': average_embedding.clone().detach()}

    @torch.no_grad()
    def eval_model(self, mode='test'):
        self.model.eval()
        target, pred, loss = [], [], []
        with torch.no_grad():
            for _, batch in enumerate(self.loader.pa_loader):
                batch = batch.cuda(self.gpu_id)
                if mode == 'test':
                    mask = batch.test_mask
                elif mode == 'valid': 
                    mask = batch.val_mask
                elif mode == 'train':
                    mask = batch.train_mask
                else :
                    print('error wrong mode')

                out = self.model(batch)

                if torch.sum(mask).item() == 0: 
                    lss = 0.0
                else: 
                    lss = F.cross_entropy(out[mask], batch.y[mask])
                    loss.append(lss.detach().cpu())

                pred.append(out[mask])
                target.append(batch.y[mask])

            if len(target) > 0: 
                stacked_pred = torch.stack(pred).view(-1, self.args.n_classes)
                stacked_target = torch.stack(target).view(-1)
                preds_max = stacked_pred.max(1)[1]
                acc = preds_max.eq(stacked_target).sum().item() / stacked_target.size(0)
            else:
                acc = 1.0

            mean_loss = np.mean(loss) if loss else 0.0

        return acc, mean_loss

    def train_client_model(self, update_client_embedding): 
        delta_param = OrderedDict()
        generated_param = self.sd[self.client_id]['generated model params']
        del self.sd[self.client_id]['generated model params']

        # load generated params
        self.model.conv1.weight = nn.Parameter(generated_param['gcn1.weight'])
        self.model.conv1.bias = nn.Parameter(generated_param['gcn1.bias'])
        self.model.conv2.weight = nn.Parameter(generated_param['gcn2.weight'])
        self.model.conv2.bias = nn.Parameter(generated_param['gcn2.bias'])

        # evaluate on generated model
        with torch.no_grad():
            self.model.eval()
            val_gen_acc, val_gen_lss = self.eval_model(mode='valid')
            test_gen_acc, test_gen_lss = self.eval_model(mode='test')
            train_gen_acc, train_gen_lss = self.eval_model(mode='train')
        
        c_epoch = self.args.client_train_epochs
        
        for epoch in range(c_epoch):
            self.model.train()
            for _, batch in enumerate(self.loader.pa_loader):
                batch = batch.cuda(self.gpu_id)
                self.optimizer.zero_grad()
                out = self.model(batch)
                train_lss = F.cross_entropy(out[batch.train_mask], batch.y[batch.train_mask])
                train_lss.backward()
                self.optimizer.step()

        final_param = self.model.state_dict()

        # evaluate on trained model
        with torch.no_grad():
            self.model.eval()
            val_train_acc, val_train_lss = self.eval_model(mode='valid')
            test_train_acc, test_train_lss = self.eval_model(mode='test')
            train_train_acc, train_train_lss = self.eval_model(mode='train')

        if update_client_embedding: 
            self.model.eval()
            with torch.no_grad():
                for _, batch in enumerate(self.loader.pa_loader):
                    batch = batch.cuda(self.gpu_id)
                    embedding_tmp = self.model(batch, is_proxy=True)
        
            average_embedding = torch.mean(embedding_tmp, dim=0, keepdim=True)
            self.sd[self.client_id] = {'functional_embedding': average_embedding.clone().detach()}

            del embedding_tmp
            del average_embedding
        ################################
        # calculate delta param
        delta_param['gcn1.weight'] = final_param['conv1.weight'] - generated_param['gcn1.weight']
        delta_param['gcn1.bias'] = final_param['conv1.bias'] - generated_param['gcn1.bias']
        delta_param['gcn2.weight'] = final_param['conv2.weight'] - generated_param['gcn2.weight']
        delta_param['gcn2.bias'] = final_param['conv2.bias'] - generated_param['gcn2.bias']

        self.sd[self.client_id].update({'delta param': {k: v.clone().detach() for k, v in delta_param.items()},
                                    'train_acc': train_train_acc, 
                                    'val_acc': val_train_acc, 
                                    'test_acc': test_train_acc})

        print(f'[client{self.client_id}] rnd{self.curr_rnd}'
              f'val_acc={val_train_acc}, test_acc={test_train_acc}')

        self.save_state()

    def switch_state(self, client_id):
        self.client_id = client_id
        self.loader.switch(client_id)
        self.logger.switch(client_id)
        
        if self.is_initialized():
            time.sleep(0.1)
            self.load_state()
        else:
            self.init_state()

    def is_initialized(self):
        return os.path.exists(os.path.join(self.args.checkpt_path, f'{self.client_id}_state.pt'))

    