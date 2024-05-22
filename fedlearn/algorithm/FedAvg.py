from fedlearn.algorithm.FedBase import Server as BasicServer
from fedlearn.models.client import Client
from fedlearn.utils.metric import Metrics
from tqdm import tqdm
import time
import numpy as np
import torch

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
        self.beta = options['beta']
        self.simple_average = options['simple_average']
    
    def train(self):
        print('>>> Select {} clients for aggregation per round \n'.format(self.clients_per_round))
        if self.gpu:
            self.latest_model = self.latest_model.to(self.device)

        for self.current_round in tqdm(range(self.num_round)):
            tqdm.write('>>> Round {}, latest model.norm = {}'.format(self.current_round, self.latest_model.norm()))

            # Test latest model on train and eval data
            stats = self.test(self.clients, self.current_round)
            self.metrics.update_model_stats(self.current_round, stats)

            self.iterate()
            self.current_round += 1

        # Test final model on train data
        self.stats = self.test(self.clients, self.current_round)
        self.metrics.update_model_stats(self.num_round, self.stats)

        # Save tracked information
        self.metrics.write()

    def iterate(self):
        selected_clients = self.select_clients(self.clients, self.current_round)

        # Do local update for the selected clients
        solns, stats = [], []
        for c in selected_clients:
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
                    # print(num_sample)
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
    