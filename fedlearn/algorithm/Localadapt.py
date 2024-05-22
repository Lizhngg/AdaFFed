from fedlearn.algorithm.FedBase import Server as BasicServer
from fedlearn.models.client import Client
from fedlearn.utils.metric import Metrics
from fedlearn.utils.model_utils import get_sort_idxs, get_cdf, get_sample_target
import copy
import random
from tqdm import tqdm
import time
import numpy as np
from torch import nn
import torch
import cvxpy as cp

class Server(BasicServer):
    def __init__(self, dataset, options, name=''):
        super().__init__(dataset, options, name)
        self.clients = self.setup_clients(Client, dataset, options)
        assert len(self.clients) > 0
        print('>>> Initialize {} clients in total'.format(len(self.clients)))

        # Initialize system metrics
        self.name = '_'.join([name, f'wn{self.clients_per_round}', f'tn{len(self.clients)}'])
        self.metrics = Metrics(self.clients, options)
        self.aggr = options['aggr']
        self.wasserstein_beta = options['w_beta']
        self.beta = options['beta']
        self.simple_average = options['simple_average']
        self.a_value = options['a_value']
        self.c_value = options['c_value']
        self.eta = options['eta']
        # self.num_OT_local_round = options['num_OT_local_round']
    
    def train(self):
        if self.gpu:
            self.latest_model = self.latest_model.to(self.device)

        a_value = self.a_value
        c_value = self.c_value
        eta = self.eta
        loss_g = []

        for c in self.clients:
            c.w0 = torch.ones((c.train_data.X.shape[0]))
            c.w1 = c.w0 + 10000
        
        w0_g = torch.concatenate([c.w0.clone() for c in self.clients])
        w1_g = w0_g + 10000

        for self.current_round in tqdm(range(self.num_round)):
            tqdm.write('>>> Round {}, latest model.norm = {}'.format(self.current_round, self.latest_model.norm()))

            # Test latest model on train and eval data
            stats = self.test(self.clients, self.current_round)
            self.metrics.update_model_stats(self.current_round, stats)
        
            if self.dif(w0_g, w1_g) < 0.001:
                break
            else: 
                w1_g = w0_g
                solns, stats = [], []
                # print(self.latest_model)
                for c in self.clients:
                    c.set_params(self.latest_model)
                    pred_score, predicted, c_A = c.get_pred()
                    criterion_proba = nn.BCELoss(reduction='none')
                    c.loss = criterion_proba(pred_score, torch.tensor(c.train_data.Y).reshape(-1,1))
                    for (Y,A) in [(1,1), (1,0), (0,1), (0,0)]:
                        loss_c = c.loss[(c.train_data.Y==Y) & (c.train_data.A==A)]
                        idx_c = torch.from_numpy((c.train_data.Y==Y) & (c.train_data.A==A)).squeeze()
                        if len(loss_c) == 0:
                            continue
                        else:
                            w0_group = torch.tensor(self.optim(loss_c, a_value, c_value * len(loss_c) ), dtype=torch.float32)
                            c.w0[idx_c] = (1 - eta) * c.w0[idx_c] + eta * w0_group
                    
                    soln, stat = c.adapt_update(c.w0, adapt_round=10)
                        # soln, stat = c.local_train()

                    if self.print:
                        tqdm.write('>>> Round: {: >2d} local acc | CID:{}| loss {:>.4f} | Acc {:>5.2f}% | Time: {:>.2f}s'.format(
                            self.current_round, c.cid, stat['loss'], stat['acc'] * 100, stat['time']
                            ))
                            
                        # Add solutions and stats
                    solns.append(soln)
                    stats.append(stat)

                self.latest_model = self.aggregate(solns, seed = self.current_round, stats = stats)
                self.stats = self.test(self.clients, self.current_round)
                # self.current_round += 1

                self.current_round += 1
                w0_g = torch.concatenate([c.w0.clone() for c in self.clients])
        # Test final model on train data
        self.stats = self.test(self.clients, self.current_round)
        self.metrics.update_model_stats(self.num_round, self.stats)

        # Save tracked information
        self.metrics.write()

    def iterate(self):
        # selected_clients = self.select_clients(self.clients, self.current_round)

        # Do local update for the selected clients
        solns, stats = [], []
        for c in self.clients:
            # Communicate the latest global model
            c.set_params(self.latest_model)

            # Solve local and personal minimization
            soln, stat = c.local_train()
            
            if self.print:
                tqdm.write('>>> Round: {: >2d} local acc | CID:{}| loss {:>.4f} | Acc {:>5.2f}% | Time: {:>.2f}s'.format(
                    self.current_round, c.cid, stat['loss'], stat['acc'] * 100, stat['time']
                    ))
                        
            # Add solutions and stats
            solns.append(soln)
            stats.append(stat)
    
        self.latest_model = self.aggregate(solns, seed = self.current_round, stats = stats)
        return True

    def aggregate(self, solns, seed, stats):
        averaged_solution = torch.zeros_like(self.latest_model)

        num_samples, chosen_solns = [info[0] for info in solns], [info[1] for info in solns]
        if self.aggr == 'mean':  
            if self.simple_average:
                num = 0
                for num_sample, local_soln in zip(num_samples, chosen_solns):
                    num += 1
                    averaged_solution += local_soln
                averaged_solution /= num
            else:      
                selected_sample = 0
                for num_sample, local_soln in zip(num_samples, chosen_solns):
                    averaged_solution += num_sample * local_soln
                    selected_sample += num_sample
                    # print("local_soln:{},num_sample:{}".format(local_soln, num_sample))
                averaged_solution = averaged_solution / selected_sample
                # print(averaged_solution)

        elif self.aggr == 'median':
            stack_solution = torch.stack(chosen_solns)
            averaged_solution = torch.median(stack_solution, dim = 0)[0]
        elif self.aggr == 'krum':
            f = int(len(chosen_solns) * 0)
            dists = torch.zeros(len(chosen_solns), len(chosen_solns))
            scores = torch.zeros(len(chosen_solns))
            for i in range(len(chosen_solns)):
                for j in range(i, len(chosen_solns)):
                    dists[i][j] = torch.norm(chosen_solns[i] - chosen_solns[j], p = 2)
                    dists[j][i] = dists[i][j]
            for i in range(len(chosen_solns)):
                d = dists[i]
                d, _ = d.sort()
                scores[i] = d[:len(chosen_solns) - f - 1].sum()
            averaged_solution = chosen_solns[torch.argmin(scores).item()]
                
        averaged_solution = (1 - self.beta) * self.latest_model + self.beta * averaged_solution
        return averaged_solution.detach()


    def optim(self, loss, a, c):
        A = loss
        x = cp.Variable(loss.shape[0])
        objective = cp.Maximize(-a*cp.sum_squares(x)+cp.sum(cp.multiply(A,x)))
        constraints = [0 <= x, cp.sum(x) == c]
        prob = cp.Problem(objective, constraints)   
        result = prob.solve()
        for i in range(x.value.shape[0]):
            if abs(x.value[i]) < 0.01 or x.value[i] < 0:
                x.value[i] = 0
        x.value = x.value
        return x.value
    
    def dif(self, a, b):
        sum = 0
        for i in range(len(a)):
            sum += (a[i] - b[i]) ** 2
        sum0 = sum ** 0.5
        return sum0