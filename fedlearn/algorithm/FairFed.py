from fedlearn.algorithm.FedBase import Server as BasicServer
from fedlearn.models.client import Client
from fedlearn.utils.metric import Metrics 
import time
import numpy as np
import torch
import copy
from tqdm import tqdm
from torch.utils.data import DataLoader


class Server(BasicServer):
    def __init__(self, dataset, options, name=''):
        super().__init__(dataset, options, name)
        self.clients = self.setup_clients(Client, dataset, options)
        assert len(self.clients) > 0
        print('>>> Initialize {} clients in total'.format(len(self.clients)))

        # Initialize system metrics
        self.name = '_'.join([name, f'wn{self.clients_per_round}', f'tn{len(self.clients)}'])
        self.metrics = Metrics(self.clients, options)
        self.fairness = options['fair_measure']
        self.aggr = options['aggr']
        self.beta = options['fairfed_beta']

        self.fair_agg_stats = self.get_fair_stat(dataset, self.fairness)  # cpu

    def train(self):

        if self.gpu:
            self.latest_model = self.latest_model.to(self.device)

        for self.current_round in tqdm(range(self.num_round)):
            print('>>> Round {}, latest model.norm = {}'.format(self.current_round, self.latest_model.norm()))

            # Test latest model on train and eval data
            client_stats = self.test(self.clients, self.current_round)
            self.metrics.update_model_stats(self.current_round, client_stats)

            # Do local update for the selected clients
            solns, stats = [], []
            for c in self.clients:
                 # Communicate the latest global model
                c.set_params(self.latest_model)

                 # Solve local and personal minimization
                soln, stat = c.local_train()

                 # Add solutions and stats
                solns.append(soln)
                stats.append(stat)

            self.latest_model = self.aggregate(solns, seed = self.current_round, stats = stats)
        
        # Test final model on train data
        for c in self.clients: 
            c.set_params(self.latest_model)
        client_stats = self.test(self.clients, self.num_round)
        self.metrics.update_model_stats(self.num_round, client_stats)

        # Save tracked information
        self.metrics.write()

    def aggregate(self, solns, stats, seed=123):
        averaged_solution = torch.zeros_like(self.latest_model)
        num_samples, chosen_solns = [info[0] for info in solns], [info[1] for info in solns]

        fair_agg_A1, fair_agg_A0 = self.fair_agg_stats
        local_fairs = []
        fair_stats = []
        for client in self.clients:
            local_fair, (fair_A0, fair_A1, fair_coef_A0, fair_coef_A1) = self.measure_local_fair(client)
            local_fairs.append(local_fair)

            stat = fair_A0 * fair_coef_A0 / fair_agg_A0 - fair_A1 * fair_coef_A1 / fair_agg_A1
            fair_stats.append(stat)
        
        p = torch.tensor(self.train_data['num_samples']) / sum(self.train_data['num_samples'])
        fair_global = sum(p * torch.tensor(fair_stats))


        accs = torch.tensor([stat['acc'] for stat in stats])

        if self.current_round == 0:
            delta = torch.abs(accs - torch.sum(accs) / torch.mean(accs))
            self.weights = p
        else:
            delta = torch.abs(torch.tensor(local_fairs) - fair_global)

        self.weights -= self.beta * (delta - 1 / self.num_users * torch.sum(delta))
        self.weights /= torch.sum(self.weights)

        for weight, params in zip(self.weights, chosen_solns):
            averaged_solution += weight * params

        return averaged_solution.detach()

    def measure_local_fair(self, client):
        predicted = torch.tensor([])
        with torch.no_grad():
            dataloader = DataLoader(client.train_data, client.batch_size, shuffle=False)
            for i, (x,y,a) in enumerate(dataloader):
                if client.gpu:
                    x = x.to(self.device)
                pred = client.model(x).detach().cpu().squeeze()
                predicted_batch = (torch.sign(pred - 0.5) + 1) / 2
                predicted = torch.concatenate([predicted, predicted_batch])
            
            predicted = predicted.reshape(-1,1)

            if self.fairness == "EOD":
                fair_A0 = sum(predicted * (1 - client.train_data.A) * client.train_data.Y) / (sum((1 - client.train_data.A) * client.train_data.Y) + 0.001 )
                fair_A1 = sum(predicted * client.train_data.A * client.train_data.Y) / (sum(client.train_data.A * client.train_data.Y) + 0.001)
                # print(sum(client.train_data.A * client.train_data.Y))
                fair_local = fair_A0 - fair_A1
                fair_coef_A0 = sum((1 - client.train_data.A) * client.train_data.Y) / len(client.train_data)
                fair_coef_A1 = sum(client.train_data.A * client.train_data.Y) / len(client.train_data)

            return fair_local, (fair_A0, fair_A1, fair_coef_A0, fair_coef_A1)



    def get_fair_stat(self, dataset, measure):
        train_data, test_data = dataset
        fair_stats_A1 = fair_stats_A0 = 0
        if self.fairness == "EOD":
            # fair_stats = sum([sum(c_data.A * c_data.Y) for c_data in train_data['user_data'].values()]) / sum(train_data['num_samples'])
            for client, c_data in train_data['user_data'].items():
                fair_stats_A1 += sum(c_data.A * c_data.Y)
                fair_stats_A0 += sum((1 - c_data.A) * c_data.Y)
            fair_stats_A1 /= sum(train_data['num_samples'])
            fair_stats_A0 /= sum(train_data['num_samples'])
        elif self.fairness == "DP":
            for client, c_data in train_data['user_data'].items():
                fair_stats_A1 += sum(c_data.A)
                fair_stats_A0 += sum((1 - c_data.A))
            fair_stats_A1 /= sum(train_data['num_samples'])
            fair_stats_A0 /= sum(train_data['num_samples'])
        return fair_stats_A1, fair_stats_A0

