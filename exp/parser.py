# Copyright 2025 Wenfei Liang. 
# All rights reserved. 

from distutils.util import strtobool
import argparse

def str2bool(x):
    if type(x) == bool:
        return x
    elif type(x) == str:
        return bool(strtobool(x))
    else:
        raise ValueError(f'Unrecognised type {type(x)}')

def get_parser():
    parser = argparse.ArgumentParser()
    #################### general params #######################
    parser.add_argument('--model', type=str, default='fedsheaf')
    parser.add_argument('--mode', type=str, default=None, choices=['disjoint', 'overlapping'])
    parser.add_argument('--base-path', type=str, default='../') 
    parser.add_argument('--dataset', type=str, default='Cora')
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--n-clients', type=int, default=None)
    parser.add_argument('--n-rnds', type=int, default=None)
    parser.add_argument('--frac', type=float, default=1)
    parser.add_argument('--txt_no', type=int, default=0)
    parser.add_argument('--n_runs', type=int, default=1)

    ################## params for both server and client model ################
    # Model configuration
    parser.add_argument('--left_weights', dest='left_weights', type=str2bool, default=True,
                        help="Applies left linear layer")
    parser.add_argument('--right_weights', dest='right_weights', type=str2bool, default=True,
                        help="Applies right linear layer")
    parser.add_argument('--add_lp', dest='add_lp', type=str2bool, default=False,
                        help="Adds fixed high pass filter in the restriction maps")
    parser.add_argument('--add_hp', dest='add_hp', type=str2bool, default=False,
                        help="Adds fixed low pass filter in the restriction maps")
    parser.add_argument('--orth', type=str, choices=['matrix_exp', 'cayley', 'householder', 'euler'],
                        default='householder', help="Parametrisation to use for the orthogonal group.")
    parser.add_argument('--edge_weights', dest='edge_weights', type=str2bool, default=True,
                        help="Learn edge weights for connection Laplacian")
    parser.add_argument('--sparse_learner', dest='sparse_learner', type=str2bool, default=True)
    parser.add_argument('--max_t', type=float, default=1.0, help="Maximum integration time.")

    ################## params for server_model ################
    parser.add_argument('--server-model', type=str, default='DiagSheaf')
    # Optimisation params
    parser.add_argument('--server-lr', type=float, default=0.02)
    parser.add_argument('--server_weight_decay', type=float, default=0.0005)
    parser.add_argument('--server_sheaf_decay', type=float, default=0.0005)

    # Model configuration
    parser.add_argument('--server_d', type=int, default=1)
    parser.add_argument('--server_layers', type=int, default=4)
    parser.add_argument('--server_normalised', dest='server_normalised', type=str2bool, default=True,
                        help="Use a normalised Laplacian")
    parser.add_argument('--server_deg_normalised', dest='server_deg_normalised', type=str2bool, default=False,
                        help="Use a a degree-normalised Laplacian")
    parser.add_argument('--server_linear', dest='server_linear', type=str2bool, default=True,
                        help="Whether to learn a new Laplacian at each step.")
    parser.add_argument('--server_second_linear', dest='server_second_linear', type=str2bool, default=False)
    parser.add_argument('--server_hidden_channels', type=int, default=20)
    parser.add_argument('--server_input_dropout', type=float, default=0.0)
    parser.add_argument('--server_dropout', type=float, default=0.3)
    parser.add_argument('--server_use_act', dest='server_use_act', type=str2bool, default=True)
    parser.add_argument('--server_sheaf_act', type=str, default="tanh", help="Activation to use in sheaf learner.")
    
    # HN params ####
    parser.add_argument('--HN_hidden_dim', type=int, default=128)
    parser.add_argument('--server_hn_lr', type=float, default=0.01)
    parser.add_argument('--hn_dropout', type=float, default=0.3)
    
    ################## params for client_model ################
    parser.add_argument('--client-lr', type=float, default=0.02)
    parser.add_argument('--client_weight_decay', type=float, default=0.0005)
    parser.add_argument('--client_hidden_dim', type=int, default=128)
    parser.add_argument('--client_vector_epochs', type=int, default=10)
    parser.add_argument('--client_train_epochs', type=int, default=5)
    parser.add_argument('--client_dropout', type=float, default=0.3) 

    return parser