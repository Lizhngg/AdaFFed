from fedlearn.algorithm.FedBase import Server as BasicServer
from fedlearn.models.client import Client
from fedlearn.utils.metric import Metrics 
from fedlearn.utils.model_utils import aggregate
import time
import numpy as np
import torch
from tqdm import tqdm
import random

class Server(BasicServer):
    def __init__(self, dataset, options, name=''):
        super().__init__(dataset, options, name)
        print('Using agnostic flearn (non-stochastic version) to Train')

        self.clients = self.setup_clients(Client, dataset, options)
        assert len(self.clients) > 0
        print('>>> Initialize {} clients in total'.format(len(self.clients)))

        self.train_num_samples = dataset[0]['num_samples']
        self.users = dataset[0]['users']
        # Initialize system metrics
        self.learning_rate_lambda = options['weight_lr']
        self.learning_rate = options['local_lr']
        self.name = '_'.join([name, f'wn{self.clients_per_round}', f'tn{len(self.clients)}'])
        self.metrics = Metrics(self.clients, options)
        self.resulting_model = self.model.get_flat_params()
        self.options = options

    def train(self):
        if self.gpu:
            self.latest_model = self.latest_model.to(self.device)

        # self.dynamic_weight = np.ones(self.num_users) * 1.0 / self.num_users 
        self.dynamic_weight = np.array(self.train_num_samples)/sum(self.train_num_samples)

        for self.current_round in tqdm(range(self.num_round+1)):
            tqdm.write(f'>>> Round {self.current_round}, latest model.norm = {self.latest_model.norm()}')
            tqdm.write(f"lambda:{self.dynamic_weight}")
            
            # Test latest model on train and eval data
            stats = self.test(self.clients,self.current_round)
            self.metrics.update_model_stats(self.current_round, stats)

            solns, losses = [], []

            for idx, c in enumerate(self.clients):
                c.set_params(self.latest_model)
                # batch_idx = random.sample(range(len(c.train_data)), 512)
                # batch_all = [torch.tensor(c.train_data.X)[batch_idx], torch.tensor(c.train_data.Y)[batch_idx]]
                batch_all = [torch.tensor(c.train_data.X), torch.tensor(c.train_data.Y)]
                grad, loss = c.get_grads(batch_all)

                solns.append((self.dynamic_weight[idx], grad))
                losses.append(loss)

            avg_gradient = aggregate(solns)

            self.latest_model -= self.learning_rate * avg_gradient.to(self.device)

            # for v,g in zip(self.latest_model, avg_gradient):
            #     # self.latest_model is updated here.
            #     v -= self.learning_rate * g

            for idx in range(len(self.dynamic_weight)):
                self.dynamic_weight[idx] += self.learning_rate_lambda * losses[idx].cpu().detach().numpy()
            
            self.dynamic_weight = self.project([i + 0.0001 for i in self.dynamic_weight])

            self.resulting_model = (self.resulting_model * self.current_round + self.latest_model) * 1.0 / (self.current_round+1)
            # for k in range(len(self.resulting_model)):
            #     self.resulting_model[k] = (self.resulting_model[k] * self.current_round + self.latest_model[k]) * 1.0 / (self.current_round+1)

        self.latest_model = self.resulting_model

        # Test final model on train data
        stats = self.test(self.clients, self.current_round)
        self.metrics.update_model_stats(self.num_round, stats)

        # Save tracked information
        self.metrics.write()

    def project(self, y):
        ''' algorithm comes from:
        https://arxiv.org/pdf/1309.1541.pdf
        '''
        u = sorted(y, reverse=True)
        x = []
        rho = 0
        for i in range(len(y)):
            if (u[i] + (1.0/(i+1)) * (1-np.sum(np.asarray(u)[:i]))) > 0:
                rho = i + 1
        lambda_ = (1.0/rho) * (1-np.sum(np.asarray(u)[:rho]))
        for i in range(len(y)):
            x.append(max(y[i]+lambda_, 0))
        return x