# Copyright 2024 Wenfei Liang. 
# All rights reserved. 

import os
from exp.parser import get_parser
from datetime import datetime
import itertools

from misc.utils import *
from modules.mainprocs import MainProcess

def main(args):
    args = set_config(args)
   
    if args.model == 'fedsheaf':    
        from models.fedsheaf.server import Server
        from models.fedsheaf.client import Client
    else:
        print('incorrect model was given: {}'.format(args.model))
        os._exit(0)

    if args.dataset == 'Cora':
        args.gcn_layer_dims = [(1433, args.client_hidden_dim),(args.client_hidden_dim, args.client_hidden_dim)]
    elif args.dataset == 'CiteSeer':
        args.gcn_layer_dims = [(3703, args.client_hidden_dim),(args.client_hidden_dim, args.client_hidden_dim)]
    elif args.dataset == 'PubMed': 
        args.gcn_layer_dims = [(500, args.client_hidden_dim),(args.client_hidden_dim, args.client_hidden_dim)]
    elif args.dataset == 'ogbn-arxiv': 
        args.gcn_layer_dims = [(128, args.client_hidden_dim),(args.client_hidden_dim, args.client_hidden_dim)]
    elif args.dataset == 'Computers': 
        args.gcn_layer_dims = [(767, args.client_hidden_dim),(args.client_hidden_dim, args.client_hidden_dim)]
    elif args.dataset == 'Photo': 
        args.gcn_layer_dims = [(745, args.client_hidden_dim),(args.client_hidden_dim, args.client_hidden_dim)]  

    pp = MainProcess(args, Server, Client)
    best_val_acc, best_test_acc = pp.start()
    return best_val_acc, best_test_acc

def set_config(args):
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    trial = f'{args.dataset}_{args.mode}/clients_{args.n_clients}/{now}_{args.model}'

    args.data_path = f'{args.base_path}/datasets' 
    args.checkpt_path = f'{args.base_path}/checkpoints/{trial}'
    args.log_path = f'{args.base_path}/logs/{trial}'

    return args

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()

    n_runs = args.n_runs
    
    s_weightdecay = [0.0005] 
    s_HN_hidden_dim = [128]
    c_weightdecay = [0.0005] 
    c_hidden_dim = [128]

    s_lr = [0.001]
    s_hn_lr = [0.01]
    c_lr = [0.01]
   
    best_result_info = {
        "path": None,
        "best_test_result": 0,
        "test_std": 0,
        "server_lr": None,
        "server_weight_decay": None,
        "server_sheaf_decay": None,
        "server_HN_hidden_dim": None, 
        "server_hn_lr": None,
        "client_lr": None,
        "client_weight_decay": None,
        "client_hidden_dim": None, 
        "client_train_epochs": 0
        }
    
    param_result = {
        "path": None,
        "avg_test_result": 0,
        "test_std": 0,
        "rnd_test_results": 0,
        "server_lr": None,
        "server_weight_decay": None,
        "server_sheaf_decay": None,
        "server_HN_hidden_dim": None, 
        "server_hn_lr": None,
        "client_lr": None,
        "client_weight_decay": None,
        "client_hidden_dim": None,
        "client_train_epochs": 0
        }
    
    filename = f'../logs/{args.dataset}_{args.mode}/clients_{args.n_clients}/params_results_{args.txt_no}.txt'

    args.server_lr = s_lr
    args.server_weight_decay = s_weight_decay
    args.server_sheaf_decay = s_weight_decay

    args.HN_hidden_dim = s_HN_hidden_dim
    args.server_hn_lr = s_hn_lr

    args.client_lr = c_lr
    args.client_weight_decay = c_weight_decay
    args.client_hidden_dim = c_hidden_dim

    args.server_dropout = server_dropout
    args.hn_dropout = hn_dropout
    args.client_dropout = client_dropout

    best_val = []
    best_test = []
    for run_no in range(n_runs): 
        args.seed = args.seed + run_no
        val_tmp, test_tmp = main(args)
        best_val.append(val_tmp)
        best_test.append(test_tmp)
        
    best_val_acc = np.mean(np.array(best_val)*100)
    best_test_acc = np.mean(np.array(best_test)*100)
    test_std = np.std(np.array(best_test)*100)
    print(f'[main] val accs: {best_val}, test accs: {best_test}, best_val_acc: {best_val_acc}, \
            best_test_acc: {best_test_acc}, test_std: {test_std}')

    param_result["path"] = args.log_path
    param_result["avg_test_result"] = best_test_acc
    param_result["test_std"] = test_std
    param_result["rnd_test_results"] = best_test
    param_result["server_lr"] = args.server_lr
    param_result["server_weight_decay"] = args.server_weight_decay
    param_result["server_sheaf_decay"] = args.server_sheaf_decay
    param_result["server_HN_hidden_dim"] = args.HN_hidden_dim
    param_result["server_hn_lr"] = args.server_hn_lr
    param_result["client_lr"] = args.client_lr
    param_result["client_weight_decay"] = args.client_weight_decay
    param_result["client_hidden_dim"] = args.client_hidden_dim
    param_result["client_train_epochs"] = args.client_train_epochs
    param_result["client_dropout"] = args.client_dropout
    param_result["server_dropout"] = args.server_dropout
    param_result["hn_dropout"] = args.hn_dropout

    with open(filename, 'a') as f:
        json.dump(param_result, f, indent=2)
        f.write("\n")

    if best_test_acc > best_result_info["best_test_result"]:
        best_result_info["path"] = args.log_path
        best_result_info["best_test_result"] = best_test_acc
        best_result_info["test_std"] = test_std
        best_result_info["server_lr"] = args.server_lr
        best_result_info["server_weight_decay"] = args.server_weight_decay
        best_result_info["server_sheaf_decay"] = args.server_sheaf_decay
        best_result_info["server_HN_hidden_dim"] = args.HN_hidden_dim
        best_result_info["server_hn_lr"] = args.server_hn_lr
        best_result_info["client_lr"] = args.client_lr
        best_result_info["client_weight_decay"] = args.client_weight_decay
        best_result_info["client_hidden_dim"] = args.client_hidden_dim
        best_result_info["client_train_epochs"] = args.client_train_epochs
        best_result_info["client_dropout"] = args.client_dropout
        best_result_info["server_dropout"] = args.server_dropout
        best_result_info["hn_dropout"] = args.hn_dropout

        with open(filename, 'a') as f:
            json.dump(best_result_info,f,indent=2)
            f.write("\n")










