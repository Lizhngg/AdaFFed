import numpy as np
import time
import torch
import torch.nn as nn
import torch.optim as optim
from fedlearn.models.models import choose_model
from torch.utils.data import DataLoader
from fedlearn.models.FairBatchSampler import FairBatch, CustomDataset
from fedlearn.utils.model_utils import weighted_loss
from fedlearn.utils.model_utils import get_sort_idxs, get_cdf, get_sample_target
import torch.nn.functional as F
import copy

class Client(object):

    def __init__(self, cid, train_data, test_data, options={}, model=None):

        # load params
        self.cid = cid
        self.local_lr = options['local_lr']
        self.wd = options['wd']
        self.sensitive_attr = options['sensitive_attr']
        self.batch_size = options['batch_size']
        self.data_info = options['data_info']

        # load data
        self.train_data, self.test_data = train_data, test_data
        self.A = self.train_data.A
        self.batch_size = options['batch_size']


        # initilaize local model
        self.model = model
        self.local_params = self.global_params = self.get_model_params()
        self.local_model_bytes = 0

        if options['criterion'] == 'celoss':
            self.criterion = nn.CrossEntropyLoss()
            self.mission = 'multiclass'
        elif options['criterion'] == 'mseloss':
            self.criterion = nn.MSELoss()
            self.mission = 'reg'
        elif options['criterion'] == 'bceloss':
            self.criterion = nn.BCELoss()
            self.mission = 'binary'
        self.num_local_round = options['num_local_round']

        # use gpu
        self.gpu = options['gpu'] if 'gpu' in options else False
        self.device = options['device']
        
        if 'gpu' in options and (options['gpu'] is True):
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            self.model = self.model.to(self.device)
            # print('>>> Use gpu on self.device {}'.format(self.device.index))

        if options['algorithm'] == 'lfb':
            sampler = FairBatch(self.model, torch.tensor(self.train_data.X).to(self.device), torch.tensor(self.train_data.Y).reshape(-1).to(self.device), torch.tensor(self.train_data.A).reshape(-1).to(self.device), batch_size=self.batch_size, 
                                alpha = 0.005, target_fairness = 'dp', replacement = False)
            self.train_dataloader = DataLoader(self.train_data, sampler=sampler, num_workers=0)
            self.train_dataloader_iter = enumerate(self.train_dataloader)
            self.test_dataloader = DataLoader(test_data, batch_size = self.batch_size, shuffle = False)
        elif self.batch_size == 0:
            self.train_dataloader = DataLoader(train_data, batch_size = len(train_data), shuffle = True)
            self.train_dataloader_iter = enumerate(self.train_dataloader)
            self.test_dataloader = DataLoader(test_data, batch_size = len(train_data), shuffle = False)
        else:
            self.train_dataloader = DataLoader(train_data, batch_size = self.batch_size, shuffle = True)
            self.train_dataloader_iter = enumerate(self.train_dataloader)
            self.test_dataloader = DataLoader(test_data, batch_size = self.batch_size, shuffle = False)

        self.train_samples_num = len(self.train_data)
        self.test_samples_num = len(self.test_data)

        # optimizer
        if options['local_optimizer'].lower() == 'sgd':
            self.optimizer = optim.SGD(self.model.parameters(), lr=self.local_lr, weight_decay=self.wd)
        elif options['local_optimizer'].lower() == 'adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.local_lr, weight_decay=self.wd)
        elif options['local_optimizer'].lower() == 'adagrad':
            self.optimizer = optim.adagrad(self.model.parameters(), lr=self.local_lr, weight_decay=self.wd)
        # # self.optimizer = grad_desc(self.model.parameters(), lr = options['local_lr'])
        # # torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=50, eta_min=0.0001, last_epoch=-1)
        # self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.95, last_epoch=-1)
        # # torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.8, last_epoch=-1)


    @staticmethod
    def move_model_to_gpu(model, options):
        if 'gpu' in options and (options['gpu'] is True):
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") if 'device' not in options else options['device']
            # torch.backends.cudnn.deterministic = True
            # torch.backends.cudnn.enabled = True
            # torch.backends.cudnn.benchmark = True
            model.to(device)
            print('>>> Use gpu on self.device {}'.format(device.index))
        else:
            print('>>> Do not use gpu')

    def set_params(self, flat_params):
        '''set model parameters, where input is a flat parameter'''
        self.model.set_params(flat_params)

    def get_model_params(self):
        '''get local flat model parameters, transform torch model parameters into flat tensor'''
        return self.model.get_flat_params()
    
    def get_global_params(self, global_params):
        self.global_params = copy.deepcopy(global_params)
        
    def get_grads(self, mini_batch_data):
        '''get model gradient'''
        x, y = mini_batch_data
        self.model.train()
        if self.gpu:
            x, y = x.to(self.device), y.to(self.device)
        self.optimizer.zero_grad()
        pred = self.model(x)
        loss = self.criterion(pred, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 50)
        flat_grads = self.model.get_flat_grads().cpu().detach()
        # self.optimizer.zero_grad()
        return torch.empty_like(flat_grads).copy_(flat_grads), loss.cpu().detach()
    
    # def get_pred(self):
    #     self.model.eval()
    #     with torch.no_grad():
    #         (x, y, a) = torch.tensor(self.train_data.X), torch.tensor(self.train_data.Y), torch.tensor(self.train_data.A)
    #         if self.gpu:
    #             x, y = x.to(self.device), y.to(self.device)
    #         self.optimizer.zero_grad()
    #         pred_score = self.model(x).detach().cpu()
    #         if self.mission == 'binary':
    #             predicted = ((torch.sign(pred_score - 0.5) + 1) / 2)
    #         elif self.mission == 'multiclass':
    #             _, predicted = torch.max(pred_score, 1)
    #     return pred_score.clone(), predicted.clone(), a.clone()

    def get_pred(self):
        self.model.eval()
        dataloader = DataLoader(self.train_data, batch_size = self.batch_size, shuffle = False)
        pred_score = predicted = torch.tensor([])
        with torch.no_grad():
            for i, (x, y, a) in enumerate(dataloader):
                if self.gpu:
                    x, y = x.to(self.device), y.to(self.device)
                self.optimizer.zero_grad()
                pred_score_batch = self.model(x).detach().cpu()
                if self.mission == 'binary':
                    predicted_batch = ((torch.sign(pred_score_batch - 0.5) + 1) / 2)
                elif self.mission == 'multiclass':
                    _, predicted_batch = torch.max(pred_score_batch, 1)
                pred_score = torch.cat([pred_score, pred_score_batch.squeeze()])
                predicted = torch.cat([predicted, predicted_batch.squeeze()])
        return pred_score.reshape(-1,1).clone(), predicted.reshape(-1,1).clone(), self.train_data.A
    
    def soln_sgd(self, batch_data):
        x, y = batch_data
        if self.gpu:
            x, y = x.to(self.device), y.to(self.device)
        self.model.train()
        self.optimizer.zero_grad()
        pred = self.model(x)
        loss = self.criterion(pred, y)
        loss.backward()
        self.optimizer.step()
        grad = self.model.get_flat_grads()
        params = self.model.get_flat_params()
        return params, grad

    def get_next_train_batch(self):
        try:
            _, batch_data = self.train_dataloader_iter.__next__()
        except StopIteration:
            self.train_dataloader_iter = enumerate(self.train_dataloader)
            _, batch_data = self.train_dataloader_iter.__next__()

        if self.sensitive_attr:
            (X, Y, A) = batch_data
        else:
            (X, Y) = batch_data

        return (X, Y, A)
    
    def local_wb_train(self, alpha):

        begin_time = time.time()

        # Barycenter calculation
        pred_score, predicted, c_A = self.get_pred()
        num_samples = len(self.train_data)
        A_score_function, A_len = {attr:pred_score[self.train_data.A == attr].ravel() for attr in np.unique(self.train_data.A)}, {attr:torch.sum(torch.tensor(self.train_data.A.ravel() == attr)) for attr in np.unique(self.train_data.A)}
        distributions = {Sa: get_cdf(A_score_function[Sa].ravel()) for Sa in A_score_function}
        BC_cdf = torch.stack([distributions[Sa] * (A_len[Sa] / num_samples) for Sa in distributions]).sum(axis=0)

        # T calculation
        target = {}
        tar_position = torch.zeros_like(torch.tensor(self.train_data.A))
        for attr in A_score_function:
            dsort = get_sample_target(BC_cdf, A_len[attr])
            value, index = torch.sort(A_score_function[attr])
            target[attr] = dsort[index] * 1/100
            tar_position[self.train_data.A.ravel() == attr] = target[attr].reshape(-1,1)
        self.train_data.A = torch.cat((torch.tensor(self.train_data.A), tar_position.reshape(-1,1)), dim=1)


        self.train_dataloader = DataLoader(self.train_data, batch_size = self.batch_size, shuffle = True)
        self.train_dataloader_iter = enumerate(self.train_dataloader)

        for _ in range(self.num_local_round):
            self.model.train()
            (x, y, a) = self.get_next_train_batch()
            if self.gpu:
                x, y, a = x.to(self.device), y.to(self.device), a.to(self.device)
            self.optimizer.zero_grad()
            pred = self.model(x)
            loss =  alpha * self.criterion(pred, y) + (1 - alpha) * torch.sum((pred.reshape(-1) - a[:,1])**2) / self.batch_size
            # loss = self.criterion(pred, y)
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 60)
            self.optimizer.step()

        self.local_model = self.get_model_params()
        self.train_data.A = self.A

        train_stats = self.model_eval(self.train_dataloader)
        param_dict = {'norm': torch.norm(self.local_model).item(),
            'max': self.local_model.max().item(),
            'min': self.local_model.min().item()}
        
        return_dict = {'loss': train_stats['loss'] / train_stats['num'],
            'acc': train_stats['acc'] / train_stats['num']}
        return_dict.update(param_dict)

        end_time = time.time()
        stats = {'id': self.cid, 'time': round(end_time - begin_time, 2)}
        stats.update(return_dict)
        return (len(self.train_data), self.local_model), stats


    def local_FB_train(self,alpha, lbd=None, m_yz=None):

        begin_time = time.time()

        epoch_loss = []
        nc = 0
        if lbd == None:
            m_yz, lbd = {}, {}
        for y in [0,1]:
            for z in range(len(self.set_z)):
                m_yz[(y,z)] = ((self.train_data.Y == y) & (self.train_data.A == z)).sum()

        for y in [0,1]:
            for z in range(len(self.set_z)):
                lbd[(y,z)] = m_yz[(y,z)]/(m_yz[(0,z)] + m_yz[(1,z)])

        for i in range(self.num_local_round):
            batch_loss = []
            self.model.train()
            for batch_idx, (X, Y, A) in enumerate(self.train_dataloader):
                X, Y = X.to(self.device), Y.to(self.device)
                A = A.to(self.device)
                logits = self.model(X)

                v = torch.ones(len(Y)).type(torch.DoubleTensor).to(self.device)
                
                group_idx = {}
                for y, z in lbd:
                    group_idx[(y,z)] = torch.where((Y == y) & (A == z))[0].cpu()
                    v[group_idx[(y,z)]] = lbd[(y,z)] / (m_yz[(1,z)] + m_yz[(0,z)])
                    nc += v[group_idx[(y,z)]].sum().item()

                loss = weighted_loss(logits, Y, v, False)

                self.optimizer.zero_grad()
                if not np.isnan(loss.item()): loss.backward()
                self.optimizer.step()

                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))

        self.model.eval()

        _, _, loss_yz = self.FB_inference(self.set_z, train=True)

        for y, z in loss_yz:
                loss_yz[(y,z)] = loss_yz[(y,z)]/(m_yz[(0,z)] + m_yz[(1,z)])

        for z in range(len(self.set_z)):
            if z == 0:
                lbd[(0,z)] -= alpha ** .5 * sum([(loss_yz[(0,0)] + loss_yz[(1,0)] - loss_yz[(0,z)] - loss_yz[(1,z)]) for z in range(len(self.set_z))])
                lbd[(0,z)] = lbd[(0,z)].item()
                lbd[(0,z)] = max(0, min(lbd[(0,z)], 2*(m_yz[(1,0)]+m_yz[(0,0)])/len(self.train_data)))
                lbd[(1,z)] = 2*(m_yz[(1,0)]+m_yz[(0,0)])/len(self.train_data) - lbd[(0,z)]
            else:
                lbd[(0,z)] += alpha ** .5 * (loss_yz[(0,0)] + loss_yz[(1,0)] - loss_yz[(0,z)] - loss_yz[(1,z)])
                lbd[(0,z)] = lbd[(0,z)].item()
                lbd[(0,z)] = max(0, min(lbd[(0,z)], 2*(m_yz[(1,0)]+m_yz[(0,0)])/len(self.train_data)))
                lbd[(1,z)] = 2*(m_yz[(1,0)]+m_yz[(0,0)])/len(self.train_data) - lbd[(0,z)]

        self.local_model = self.get_model_params()

        train_stats = self.model_eval(self.train_dataloader)
        param_dict = {'norm': torch.norm(self.local_model).item(),
            'max': self.local_model.max().item(),
            'min': self.local_model.min().item()}
        
        return_dict = {'loss': train_stats['loss'] / train_stats['num'],
            'acc': train_stats['acc'] / train_stats['num']}
        return_dict.update(param_dict)

        end_time = time.time()
        stats = {'id': self.cid, 'time': round(end_time - begin_time, 2)}
        stats.update(return_dict)

        # weight, loss
        return (len(self.train_data), self.local_model), stats, sum(epoch_loss) / len(epoch_loss), nc, lbd, m_yz
    
    def local_fb(self):

        begin_time = time.time()

        for _ in range(self.num_local_round):
            self.model.train()
            (x, y, A) = self.get_next_train_batch()
            if self.gpu:
                x, y = torch.squeeze(x.to(self.device)), y.to(self.device).reshape(-1,1)
            self.optimizer.zero_grad()
            pred = self.model(x)
            loss = self.criterion(pred, y)
            loss.backward()
            self.optimizer.step() 
        
        self.local_model = self.get_model_params()

        train_stats = self.model_eval(self.train_dataloader)
        param_dict = {'norm': torch.norm(self.local_model).item(),
            'max': self.local_model.max().item(),
            'min': self.local_model.min().item()}
        
        return_dict = {'loss': train_stats['loss'] / train_stats['num'],
            'acc': train_stats['acc'] / train_stats['num']}
        return_dict.update(param_dict)

        end_time = time.time()
        stats = {'id': self.cid, 'time': round(end_time - begin_time, 2)}
        stats.update(return_dict)
        return (len(self.train_data), self.local_model), stats
    
    def adapt_update(self, w=None, last_model=None, adapt_round=10):

        assert w is None or len(w) == len(self.train_data)

        train_dataloader = DataLoader(self.train_data, batch_size = self.batch_size, shuffle = False)
        weight_dataloader = DataLoader(w, batch_size = self.batch_size, shuffle = False)
        # self.train_dataloader_iter = enumerate(zip(self.train_dataloader, self.weight_dataloader))
        begin_time = time.time()

        for _ in range(adapt_round):
            self.model.train()
            # for i, ((x, y, a), weight) in enumerate(zip(self.train_dataloader, self.weight_dataloader)):
            for i, ((x, y, a), weight) in enumerate(zip(train_dataloader, weight_dataloader)):
                if self.gpu:
                    x, y, a, weight = x.to(self.device), y.to(self.device), a.to(self.device), weight.to(self.device)
                self.optimizer.zero_grad()
                criterion = nn.BCELoss() if w is None else nn.BCELoss(weight=weight.clone().reshape(-1,1))
                pred = self.model(x)
                if last_model is not None:
                    loss = criterion(pred, y) + 0.001 * torch.norm(torch.cat([params.view(-1) for params in self.model.parameters()]) - last_model, p=2)
                else:
                    loss = criterion(pred, y)
                loss.backward()
                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 60)
                self.optimizer.step()

        # print(f"\n learnig rate: {self.optimizer.param_groups[0]['lr']}")
        # self.scheduler.step()

        self.local_model = self.get_model_params()
        self.train_data.A = self.A

        train_stats = self.model_eval(self.train_dataloader)
        # print(train_stats)
        param_dict = {'norm': torch.norm(self.local_model).item(),
            'max': self.local_model.max().item(),
            'min': self.local_model.min().item()}
        
        return_dict = {'loss': train_stats['loss'] / train_stats['num'],
            'acc': train_stats['acc'] / train_stats['num']}
        return_dict.update(param_dict)

        end_time = time.time()
        stats = {'id': self.cid, 'time': round(end_time - begin_time, 2)}
        stats.update(return_dict)
        return (len(self.train_data), self.local_model), stats

    def ot_new_update(self, alpha, ot_round):

        self.train_dataloader = DataLoader(self.train_data, batch_size = self.batch_size, shuffle = True)
        self.train_dataloader_iter = enumerate(self.train_dataloader)

        begin_time = time.time()

        for _ in range(ot_round):
            self.model.train()
            (x, y, a) = self.get_next_train_batch()
            if self.gpu:
                x, y, a = x.to(self.device), y.to(self.device), a.to(self.device)
            self.optimizer.zero_grad()
            pred = self.model(x)
            loss =  alpha * self.criterion(pred, y) + (1 - alpha) * torch.sum(torch.abs(pred.reshape(-1) - a[:,1])) / len(y)
            # loss = self.criterion(pred, y)
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 60)
            self.optimizer.step()
        # print(f"\n learnig rate: {self.optimizer.param_groups[0]['lr']}")
        # self.scheduler.step()

        self.local_model = self.get_model_params()
        self.train_data.A = self.A

        train_stats = self.model_eval(self.train_dataloader)
        param_dict = {'norm': torch.norm(self.local_model).item(),
            'max': self.local_model.max().item(),
            'min': self.local_model.min().item()}
        
        return_dict = {'loss': train_stats['loss'] / train_stats['num'],
            'acc': train_stats['acc'] / train_stats['num']}
        return_dict.update(param_dict)

        end_time = time.time()
        stats = {'id': self.cid, 'time': round(end_time - begin_time, 2)}
        stats.update(return_dict)
        return (len(self.train_data), self.local_model), stats
    
    def local_train(self):
        bytes_w = self.local_model_bytes

        begin_time = time.time()

        for _ in range(self.num_local_round):
            self.model.train()
            (x, y, a) = self.get_next_train_batch()
            if self.gpu:
                x, y = torch.squeeze(x.to(self.device)), y.to(self.device).reshape(-1,1)
            # for batch_idx, batch_data in enumerate(self.train_dataloader):
            #     if self.sensitive_attr:
            #         (x, y, A) = batch_data
            #     else:
            #         (x, y) = batch_data
            #     if self.gpu:
            #         x, y = x.to(self.device), y.to(self.device)
            self.optimizer.zero_grad()
            pred = self.model(x)
            loss = self.criterion(pred, y)
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 60)
            self.optimizer.step() 
        # print(f"\n learnig rate: {self.optimizer.param_groups[0]['lr']}")
        # self.scheduler.step()

        self.local_model = self.get_model_params()

        train_stats = self.model_eval(self.train_dataloader)
        param_dict = {'norm': torch.norm(self.local_model).item(),
            'max': self.local_model.max().item(),
            'min': self.local_model.min().item()}
        
        return_dict = {'loss': train_stats['loss'] / train_stats['num'],
            'acc': train_stats['acc'] / train_stats['num']}
        return_dict.update(param_dict)

        end_time = time.time()
        stats = {'id': self.cid, 'time': round(end_time - begin_time, 2)}
        stats.update(return_dict)
        return (len(self.train_data), self.local_model), stats

    def FB_train(self):

        begin_time = time.time()

        epoch_loss = []
        nc = 0

        for _ in range(self.num_local_round):
            batch_loss = []
            self.model.train()
            for batch_idx, batch_data in enumerate(self.train_dataloader):
                (X, Y, A) = batch_data
                if self.gpu:
                    X, Y, A = X.to(self.device), Y.to(self.device), A.to(self.device)
                pred = self.model(X)

                v = torch.ones(len(Y)).type(torch.DoubleTensor).to(self.device)

                group_idx = {}
                for y, z in self.lbd:
                    group_idx[(y,z)] = torch.where((Y == y) & (A == z))[0].cpu()
                    v[group_idx[(y,z)]] = self.lbd[(y,z)] / (self.m_yz[(1,z)] + self.m_yz[(0,z)])
                    nc += v[group_idx[(y,z)]].sum().item()

                loss = weighted_loss(pred, Y, v, False)

                self.optimizer.zero_grad()
                if not np.isnan(loss.item()): loss.backward()
                self.optimizer.step()

                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))

        self.local_model = self.get_model_params()

        train_stats = self.model_eval(self.train_dataloader)
        param_dict = {'norm': torch.norm(self.local_model).item(),
            'max': self.local_model.max().item(),
            'min': self.local_model.min().item()}
        
        return_dict = {'loss': train_stats['loss'] / train_stats['num'],
            'acc': train_stats['acc'] / train_stats['num']}
        return_dict.update(param_dict)

        end_time = time.time()
        stats = {'id': self.cid, 'time': round(end_time - begin_time, 2)}
        stats.update(return_dict)
        return (len(self.train_data), self.local_model), stats, sum(epoch_loss) / len(epoch_loss), nc

    def FB_inference(self,  set_z, truem_yz=None, train=False):
        n_yz, loss_yz, m_yz, f_z = {}, {}, {}, {}
        self.model.eval()
        for y in [0,1]:
            for z in set_z:
                loss_yz[(y,z)] = 0
                n_yz[(y,z)] = 0
                m_yz[(y,z)] = 0

        dataset = self.test_dataloader if not train else self.train_dataloader
        for _, (features, labels, sensitive) in enumerate(self.test_dataloader):
            features, labels = features.to(self.device), labels
            sensitive = sensitive
            
            # Inference
            logits = self.model(features).detach().cpu()

            pred_labels = (torch.sign(logits - 0.5) + 1) / 2
            pred_labels = pred_labels.view(-1)

            group_boolean_idx = {}
            
            for yz in n_yz:
                group_boolean_idx[yz] = (labels == yz[0]) & (sensitive == yz[1])
                n_yz[yz] += torch.sum((pred_labels == yz[0]) & (sensitive == yz[1])).item()     
                m_yz[yz] += torch.sum((labels == yz[0]) & (sensitive == yz[1])).item()    
                
                # the objective function have no lagrangian term
                loss_yz_ = F.binary_cross_entropy(logits[group_boolean_idx[yz]].to(self.device), labels[group_boolean_idx[yz]].to(self.device), reduction = 'sum').detach().cpu()
                loss_yz[yz] += loss_yz_

        for z in range(1, len(set_z)):
            if not truem_yz == None:
                f_z[z] = - loss_yz[(0,0)]/(truem_yz[(0,0)] + truem_yz[(1,0)]) + loss_yz[(1,0)]/(truem_yz[(0,0)] + truem_yz[(1,0)]) + loss_yz[(0,z)]/(truem_yz[(0,z)] + truem_yz[(1,z)]) - loss_yz[(1,z)]/(truem_yz[(0,z)] + truem_yz[(1,z)]) 
        

        return n_yz, f_z, loss_yz


        
    def model_eval(self, data, w=None):
        if isinstance(data, DataLoader):
            dataLoader = data
        else: 
            dataLoader = DataLoader(data, batch_size = self.batch_size, shuffle = False)

        self.model.eval()
        test_loss = test_acc = test_num = 0.0
        with torch.no_grad():
            for batch_data in dataLoader:
                if self.sensitive_attr:
                    (x, y, A) = batch_data
                else:
                    (x, y) = batch_data
                if self.gpu:
                    x, y = x.to(self.device), y.to(self.device).reshape(-1,1)
                
                pred = self.model(x)

                criterion = self.criterion if w is None else nn.BCELoss(weight=w.clone().reshape(-1,1).to(self.device))

                loss = criterion(pred, y)

                if self.mission == 'binary':
                    predicted = (torch.sign(pred - 0.5) + 1) / 2
                elif self.mission == 'multiclass':
                    _, predicted = torch.max(pred, 1)
                    
                correct = predicted.eq(y).sum().item()
                batch_size = y.size(0)

                test_loss += loss.item() * y.size(0) # total loss, not average
                test_acc += correct # total acc, not average
                test_num += batch_size 

        test_dict = {'loss': test_loss, 'acc': test_acc, 'num': test_num}
        return test_dict

    def local_eval(self):
        train_data_test = self.model_eval(self.train_dataloader)
        test_data_test  = self.model_eval(self.test_dataloader)
        return {'train':train_data_test, 'test':test_data_test}

        