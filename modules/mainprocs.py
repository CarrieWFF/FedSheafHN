# Copyright 2024 Wenfei Liang. 
# All rights reserved. 

import os
import sys
import time
import atexit
import numpy as np
import copy

from misc.utils import *

class MainProcess:
    def __init__(self, args, Server, Client): 
        self.args = args
        self.gpus = [int(g) for g in args.gpu.split(',')]
        self.sd = {}
        self.set_seed(self.args.seed)

        self.server = Server(args, self.sd, self.gpus[0])

        self.clients = {}
        for client_id in range(self.args.n_clients):
            self.clients[client_id] = Client(self.args, self.gpus[0], self.sd, client_id)
    
    def set_seed(self, seed=123):
        random.seed(seed)
        np.random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    def start(self):
        if os.path.isdir(self.args.checkpt_path) == False:
            os.makedirs(self.args.checkpt_path)
        if os.path.isdir(self.args.log_path) == False:
            os.makedirs(self.args.log_path)
        self.n_connected = round(self.args.n_clients*self.args.frac)
        self.update_client_embedding = False

        for curr_rnd in range(self.args.n_rnds):
            self.curr_rnd = curr_rnd
            self.selected = sorted(np.random.choice(self.args.n_clients, self.n_connected, replace=False).tolist())
            if curr_rnd != 0 and curr_rnd % 5 == 0:
                self.update_client_embedding = True
            else: 
                self.update_client_embedding = False
            st = time.time()
            ############################# generate client graph ####################################
            if curr_rnd == 0:
                message_type = 'client_generate_vector_start'
                for client_id in range(self.args.n_clients):
                    client = self.clients[client_id]
                    client.switch_state(client_id)
                    client.on_receive_message(curr_rnd, message_type)
                    client.generate_vector(client_id)
                        
                self.allclients = list(range(self.args.n_clients))
                self.server.construct_graph(self.allclients, curr_rnd)
            ############################## train server model #######################################
            self.server.on_round_begin(curr_rnd)
            self.server.train_server_GNN(curr_rnd)
            self.server.train_server_HN(curr_rnd)

            for client_id in self.selected: 
                ############################# train client model ##############################
                message_type = 'client_train_on_generated_model_prams'
                client = self.clients[client_id]
                client.on_receive_message(curr_rnd, message_type)
                client.train_client_model(self.update_client_embedding)
            ############################## update server HN ###############################
            self.server.update_server_HN(self.selected)

            if self.update_client_embedding:
                self.server.construct_graph(self.allclients, curr_rnd)

            self.server.round_end(curr_rnd, self.allclients, self.selected)
            
            print(f'[main] server model GNN and HN have been trained at round {curr_rnd}, ({time.time()-st:.2f}s)')
            self.server.save_state()

        best_val_acc, best_test_acc = self.server.procs_end()

        return best_val_acc, best_test_acc
