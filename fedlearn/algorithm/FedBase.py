import numpy as np
import torch
import copy
import time
import torch.nn as nn
from tqdm import tqdm
from fedlearn.models.models import choose_model
from torch.utils.data import DataLoader
from fedlearn.utils.metric import Metrics
from fedlearn.utils.model_utils import global_EOD, global_DP


class Server(object):
    def __init__(self, dataset, options, name = ''):
        # Basic parameters
        self.gpu = options['gpu']
        self.device = options['device']
        self.all_train_data_num = 0
        self.num_users = options['num_users']
        self.num_round = options['num_round']
        self.clients_per_round = options['clients_per_round']
        self.eval_round = options['eval_round']
        self.local_lr = options['local_lr']
        self.test_local = options['test_local']
        self.num_local_round = options['num_local_round']
        self.fair_measure = options['fairness_measure']
        self.print = options['print_result']
        self.client = []
        self.data_info = options['data_info']

        self.model = choose_model(options)
        if 'gpu' in options and (options['gpu'] is True):
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            self.model = self.model.to(device)
        self.latest_model = self.model.get_flat_params()
        self.local_model_dim = self.latest_model.shape[0]
        self.algo = options['algorithm']

    def setup_clients(self, Client, dataset, options):
        self.train_data, self.test_data = dataset
        self.all_train_data_num = sum(self.train_data['num_samples'])
        self.num_users = options['num_users']
        # print(f'Number of users = {self.num_users}')

        # Load selected client
        all_clients = []
        for user in self.train_data['users']:
            c = Client(user, self.train_data['user_data'][user], self.test_data['user_data'][user], options, copy.deepcopy(self.model))
            # init client with ID, train_data, test_data, option, model, append client to list 
            all_clients.append(c)
        
        return all_clients
    
    def select_clients(self, clients, seed = None):
        """Selects num_clients clients from possible clients"""
        num_clients = min(self.clients_per_round, len(clients))
        if seed:
            np.random.seed(seed)
        return np.random.choice(clients, num_clients, replace = False).tolist()
    
    def test(self, clients, current_round):
        begin_time = time.time()
        client_stats, ids = self.local_test(clients)
        end_time = time.time()

        train_acc = sum(client_stats['train']['accs']) / sum(client_stats['train']['num_samples'])
        train_loss = sum(client_stats['train']['losses']) / sum(client_stats['train']['num_samples'])
        train_fairness = self.global_fairness(self.train_data, self.data_info)
        
        test_acc = sum(client_stats['test']['accs']) / sum(client_stats['test']['num_samples'])
        test_loss = sum(client_stats['test']['losses']) / sum(client_stats['test']['num_samples'])
        test_fairness = self.global_fairness(self.test_data, self.data_info)

        model_stats = {'train_loss':train_loss, 'train_acc':train_acc, 'train_fairness':train_fairness,
                        'test_acc':test_acc, 'test_loss':test_loss, 'test_fairness':test_fairness,
                        'time': end_time - begin_time}
        
        # if self.print:
        #     self.print_result(current_round, model_stats, client_stats)
        self.print_result(current_round, model_stats, client_stats)

        return model_stats
    
    def local_test(self, clients):
        assert self.latest_model is not None

        stats_dict = {'num_samples':[], 'accs':[], 'losses':[]}
        client_stats= {'train':copy.deepcopy(stats_dict), 
                       'test' :copy.deepcopy(stats_dict)}

        for c in clients:
            if self.test_local:
                c.set_params(c.local_params)
            else:
                c.set_params(self.latest_model)
            client_test_dict = c.local_eval()

            for dataset, test_dict in client_test_dict.items():
                # dataset is train and test
                client_stats[dataset]['num_samples'].append(test_dict['num'])
                client_stats[dataset]['accs'].append(test_dict['acc'])
                client_stats[dataset]['losses'].append(test_dict['loss'])
        ids = [c.cid for c in clients]

        return client_stats, ids
    
    def global_fairness(self, split_data, data_info):
        self.model.set_params(self.latest_model)
        fair_gap = {f:eval('global_' + f)(split_data, self.model, self.device, data_info)[0]
                 for f in self.fair_measure}
        fair_disparity= {f + ' disparity':eval('global_' + f)(split_data, self.model, self.device, data_info)[1]
                 for f in self.fair_measure}
        fair_gap.update(fair_disparity)
        return fair_gap
    
    def print_result(self, current_round, model_stats, client_stats):
        tqdm.write("\n >>> Round: {: >4d} / Acc: {:.3%} / Loss: {:.4f} / Time: {:.2f}s /  Fairness: {} ".format(
            current_round, model_stats['train_acc'], model_stats['train_loss'], model_stats['time'], model_stats['train_fairness']))
        
        for c in range(self.num_users):
            tqdm.write('>>> Client: {} / Acc: {:.3%} / Loss: {:.4f}'.format(
                        self.train_data['users'][c], 
                        client_stats['train']['accs'][c] / client_stats['train']['num_samples'][c], 
                        client_stats['train']['losses'][c] / client_stats['train']['num_samples'][c]
                      ))
        tqdm.write('=' * 102 + "\n")

        if current_round % self.eval_round == 0:
            tqdm.write("\n = Test = round: {} / acc: {:.3%} / loss: {:.4f} / Fairness: {} ".format(
                current_round, model_stats['test_acc'],model_stats['test_loss'], model_stats['test_fairness']))