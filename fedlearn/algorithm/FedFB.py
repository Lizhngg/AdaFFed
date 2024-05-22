from fedlearn.algorithm.FedBase import Server as BasicServer
from fedlearn.models.client import Client
from fedlearn.utils.metric import Metrics
from tqdm import tqdm
import time
import numpy as np
import torch
import copy

class Server(BasicServer):
    def __init__(self, dataset, options, name=''):
        super().__init__(dataset, options, name)
        self.clients = self.setup_clients(Client, dataset, options)
        assert len(self.clients) > 0
        print('>>> Initialize {} clients in total'.format(len(self.clients)))

        # Initialize system metrics
        self.data_info = options['data_info']
        self.name = '_'.join([name, f'wn{self.clients_per_round}', f'tn{len(self.clients)}'])
        self.metrics = Metrics(self.clients, options)
        self.aggr = options['aggr']
        self.beta = options['beta']
        self.simple_average = options['simple_average']
        self.set_z = [int(a) for a in self.data_info['Alabel']]
        self.train_num = self.data_info['train_num']
        self.alpha = options['FB_alpha']
    
    def get_sensitive_num(self):
        
        m_yz, lbd = {}, {}

        # the number of samples whose label is y and sensitive attribute is z
        for y in [0,1]:
            for z in self.set_z:
                for c in self.clients:
                    print(((c.train_data.Y == y) & (c.train_data.A == z)).sum())
                    m_yz[(y,z)] = m_yz[(y,z)] + ((c.train_data.Y == y) & (c.train_data.A == z)).sum() if (y,z) in m_yz else ((c.train_data.Y == y) & (c.train_data.A == z)).sum()
        
        # init lambda
        for y in [0,1]:
            for z in self.set_z:
                lbd[(y,z)] = (m_yz[(1,z)] + m_yz[(0,z)])/self.train_num
        
        return m_yz, lbd

    def train(self):
        m_yz, lbd = self.get_sensitive_num()

        print('>>> Select {} clients for aggregation per round \n'.format(self.clients_per_round))

        if self.gpu:
            self.latest_model = self.latest_model.to(self.device)

        for self.current_round in tqdm(range(self.num_round)):
            tqdm.write('>>> Round {}, latest model.norm = {}'.format(self.current_round, self.latest_model.norm()))

            # Test latest model on train and eval data
            stats = self.test(self.clients, self.current_round)
            self.metrics.update_model_stats(self.current_round, stats)

            local_losses, nc = [], []
            solns, stats = [], []

            for c in self.clients:
                c.set_params(self.latest_model)
                c.m_yz, c.lbd = m_yz, copy.deepcopy(lbd)

                soln, stat, loss, nc_ = c.FB_train()

                nc.append(nc_)
                local_losses.append(copy.deepcopy(loss))

                # Add solutions and stats
                solns.append(soln)
                stats.append(stat)

            self.latest_model = self.aggregate(solns, seed = self.current_round, stats = stats, weight=nc)

            n_yz, f_z = {}, {}
            for z in self.set_z:
                for y in [0,1]:
                    n_yz[(y,z)] = 0

            for z in range(1, len(self.set_z)):
                f_z[z] = 0

            for c in self.clients:
                c.set_params(self.latest_model)
                n_yz_c, f_z_c, _ = c.FB_inference(self.set_z, truem_yz = m_yz, train = True)

                for yz in n_yz:
                    n_yz[yz] += n_yz_c[yz]
                
                for z in range(1, len(self.set_z)):
                    f_z[z] += f_z_c[z] + m_yz[(0,0)]/(m_yz[(0,0)] + m_yz[(1,0)]) - m_yz[(0,z)]/(m_yz[(0,z)] + m_yz[(1,z)])
                    
            for z in self.set_z:
                if z == 0:
                    lbd[(0,z)] -= self.alpha / (self.current_round + 1) ** .5 * sum([f_z[z] for z in range(1, len(self.set_z))])
                    lbd[(0,z)] = lbd[(0,z)].item()
                    lbd[(0,z)] = max(0, min(lbd[(0,z)], 2*(m_yz[(1,0)]+m_yz[(0,0)])/self.train_num))
                    lbd[(1,z)] = 2*(m_yz[(1,0)]+m_yz[(0,0)])/self.train_num - lbd[(0,z)]
                else:
                    lbd[(0,z)] += self.alpha / (self.current_round + 1) ** .5 * f_z[z]
                    lbd[(0,z)] = lbd[(0,z)].item()
                    lbd[(0,z)] = max(0, min(lbd[(0,z)], 2*(m_yz[(1,0)]+m_yz[(0,0)])/self.train_num))
                    lbd[(1,z)] = 2*(m_yz[(1,0)]+m_yz[(0,0)])/self.train_num - lbd[(0,z)]

            self.current_round += 1

        # Test final model on train data
        stats = self.test(self.clients, self.current_round)
        self.metrics.update_model_stats(self.num_round, stats)

        # Save tracked information
        self.metrics.write()


    def aggregate(self, solns, seed, stats, weight=None):
        averaged_solution = torch.zeros_like(self.latest_model)

        num_samples, chosen_solns = [info[0] for info in solns], [info[1] for info in solns]
        if self.aggr == 'mean':  
            if weight:
                sum_w = sum(weight)
                for idx, (num_sample, local_soln) in enumerate(zip(num_samples, chosen_solns)):
                    averaged_solution += local_soln * weight[idx]
                averaged_solution = averaged_solution / sum_w
            elif self.simple_average:
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
    