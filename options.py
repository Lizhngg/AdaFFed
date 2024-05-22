import argparse
from config import *

def arg_parser():
    parser = argparse.ArgumentParser()

    # Algoritha
    parser.add_argument('--algorithm', help = 'name of algorithm',
                        type = str, choices = OPTIMIZERS, default = 'adaffed')
    parser.add_argument('--num_users', help = 'number of users', 
                        type = int, default = 2)
    
    # data set
    parser.add_argument('--data', help = 'name of dataset',
                        type = str, default = 'compas')

    parser.add_argument('--data setting', help = 'operations on raw datas to generate decentralized data set; input a dict',
                        type = dict, default = {'sensitive_attr': 'race', 'dirichlet':True, 'by sensitive':True, 'alpha': 0.1, 'generate': False})
    
    parser.add_argument('--seed', help = 'operations on raw datas to generate decentralized data set; input a dict',
                        type = int, default = 5)
    
    # model
    parser.add_argument('--model', help = 'name of local model;',
                        type = str, choices = MODELS, default = '1nn')
    parser.add_argument('--criterion', help = 'name of loss function',
                        type = str, choices = CRITERIA, default = 'bceloss')
    parser.add_argument('--wd', help = 'weight decay parameter;',
                        type = float, default = 1e-4)
    parser.add_argument('--gpu', help = 'use gpu (default: True)',
                        default = True, action = 'store_true')

    # federated arguments
    parser.add_argument('--server', help = 'type of server',
                        type = str, default = 'server', choices = SERVERS)
    parser.add_argument('--num_round', help = 'number of rounds to simulate',
                        type = int, default = 25)
    parser.add_argument('--eval_round', help = 'evaluate every ___ rounds',
                        type = int, default = 1)
    parser.add_argument('--clients_per_round', help = 'number of clients selected per round',
                        type = int, default = 10)
    parser.add_argument('--batch_size', help = 'batch size when clients train on data',
                        type = int, default = 128)
    parser.add_argument('--num_epoch', help = 'number of rounds for solving the personalization sub-problem when clients train on data',
                        type = int, default = 1)
    parser.add_argument('--num_local_round', help = 'number of local rounds for local update',
                        type = int, default = 10)
    parser.add_argument('--local_lr', help = 'learning rate for local update',
                        type = float, default = 0.0001)
    parser.add_argument('--local_optimizer', help = 'optimizer for local training',
                        type = str, default = 'adam')
    parser.add_argument('--test_local', help = 'if test model with local params',
                        type = bool, default = False)
    parser.add_argument('--print_result', help = 'if print the result',
                        type = bool, default = True)
    ## Fairness
    parser.add_argument('--fairness_measure', help = 'fairness measure',
                        type = list, default = ['EOD','DP'])
    
    # arguments for federated algorithm

    ## FedAvg
    parser.add_argument('--aggr', help = 'aggregation method',
                        type = str, default = 'mean')
    parser.add_argument('--simple_average', help = 'if simple average used',
                        type = str, default = False)
    parser.add_argument('--weight_aggr', help = 'weighted aggregation',
                        type = float, default = False)
    parser.add_argument('--beta', help = 'model params momentum',
                        type = float, default = 0.95)
    
    ## AFL
    parser.add_argument('--weight_lr', help = 'learning rate for dynamic weight',
                        type = float, default = 0.5)
    
    ##FairFed
    parser.add_argument('--fairfed_beta', help = 'parameter to control the aggregation',
                        type = float, default = 1)
    parser.add_argument('--fair_measure', help = 'use which fair measure',
                        type = str, default = "EOD")
    
    ## AdaFFed
    parser.add_argument('--a_value', help = 'regularization parameter a',
                        type = float, default = 5)
    parser.add_argument('--c_value', help = 'parameter c',
                        type = str, default = 100)
    parser.add_argument('--eta', help = 'eta',
                        type = str, default = 1)
    parser.add_argument('--fairness', help = 'fairness',
                        type = str, default = 'EOD')
    
    args = parser.parse_args()

    return args