import numpy as np
import time
import torch
import torch.nn as nn
import torch.optim as optim
from fedlearn.models.models import choose_model
from torch.utils.data import DataLoader
from fedlearn.models.FairBatchSampler import FairBatch, CustomDataset
from fedlearn.models.client import Client
import copy

class local_FB_Client(Client):

    def __init__(self, cid, train_data, test_data, options={}, model=None):
        super().__init__(cid, options, train_data, test_data, options, model)
        # load params
        self.wd = options['wd']
        self.sensitive_attr = options['sensitive_attr']
        self.batch_size = options['batch_size']

        # load data
        self.train_data, self.test_data = train_data, test_data
        self.A = self.train_data.A
        self.batch_size = options['batch_size']
        if self.batch_size == 0:
            self.train_dataloader = DataLoader(train_data, batch_size = len(train_data), shuffle = True)
            self.train_dataloader_iter = enumerate(self.train_dataloader)
            self.test_dataloader = DataLoader(test_data, batch_size = len(train_data), shuffle = False)
        else:
            self.train_dataloader = DataLoader(train_data, batch_size = self.batch_size, shuffle = True)
            self.train_dataloader_iter = enumerate(self.train_dataloader)
            self.test_dataloader = DataLoader(test_data, batch_size = self.batch_size, shuffle = False)

        self.train_samples_num = len(self.train_data)
        self.test_samples_num = len(self.test_data)
        
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
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            self.model = self.model.to(device)
            # print('>>> Use gpu on device {}'.format(device.index))

        # optimizer
        if options['local_optimizer'].lower() == 'sgd':
            self.optimizer = optim.SGD(self.model.parameters(), lr=self.local_lr, weight_decay=self.wd)
        elif options['local_optimizer'].lower() == 'adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.local_lr, weight_decay=self.wd)
        elif options['local_optimizer'].lower() == 'adagrad':
            self.optimizer = optim.adagrad(self.model.parameters(), lr=self.local_lr, weight_decay=self.wd)
        # self.optimizer = grad_desc(self.model.parameters(), lr = options['local_lr'])
        # torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=50, eta_min=0.0001, last_epoch=-1)
        torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.9, last_epoch=-1)

        if options['algorithm'] == 'lfb':
            self.train_data.Y[self.train_data.Y == 0] = -1
            sampler = FairBatch(self.model, torch.tensor(self.train_data.X).to(self.device), torch.tensor(self.train_data.Y).reshape(-1).to(self.device), torch.tensor(self.train_data.A).reshape(-1).to(self.device), batch_size=self.batch_size, 
                                alpha = 0.005, target_fairness = 'dp', replacement = False)
            self.train_dataloader = DataLoader(self.train_data, sampler=sampler, num_workers=0)

    @staticmethod
    def move_model_to_gpu(model, options):
        if 'gpu' in options and (options['gpu'] is True):
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") if 'device' not in options else options['device']
            # torch.backends.cudnn.deterministic = True
            # torch.backends.cudnn.enabled = True
            # torch.backends.cudnn.benchmark = True
            model.to(device)
            print('>>> Use gpu on device {}'.format(device.index))
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
    
    def get_pred(self):
        self.model.eval()
        with torch.no_grad():
            (x, y, A) = torch.tensor(self.train_data.X), torch.tensor(self.train_data.Y), torch.tensor(self.train_data.A)
            if self.gpu:
                x, y = x.to(self.device), y.to(self.device)
            self.optimizer.zero_grad()
            pred_score = self.model(x).detach().cpu()
            if self.mission == 'binary':
                predicted = ((torch.sign(pred_score - 0.5) + 1) / 2)
            elif self.mission == 'multiclass':
                _, predicted = torch.max(pred_score, 1)
        return pred_score.clone(), predicted.clone(), A.clone()
    
    def ot_group_update(self, decouple_data, alpha, ot_round):

        begin_time = time.time()
        self.model.train()
        for _ in range(ot_round):
            self.optimizer.zero_grad()
            loss_h, loss_wasserstein = 0, 0
            sample_num = 0
            for sensitive_attr in decouple_data:
                x = torch.tensor(self.train_data.X)[decouple_data[sensitive_attr][0],:]
                y = torch.tensor(self.train_data.Y)[decouple_data[sensitive_attr][0],:]
                # print(decouple_data[sensitive_attr][0].shape)
                # print(torch.unique(torch.tensor(self.train_data.A)[decouple_data[sensitive_attr][0],:]))
                # print(len(self.train_data))
                if self.gpu:
                    x, y = x.to(self.device), y.to(self.device)
                target = decouple_data[sensitive_attr][1].reshape(-1,1).to(self.device)
                pred = self.model(x)
                loss_attr = self.criterion(pred, y) * len(pred)
                sample_num += len(pred)
                loss_wasserstein_attr = torch.sum(torch.abs(pred - target)) 
                # print(len(pred), target)
                loss_h += alpha * loss_attr 
                loss_wasserstein += (1 - alpha) * loss_wasserstein_attr
                # print(torch.unique(torch.tensor(self.train_data.A)[decouple_data[sensitive_attr][0],:]))
                # print(loss_h/len(pred))
                # print(decouple_data[sensitive_attr][0].shape)
            loss = loss_h / sample_num + loss_wasserstein
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 60)
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

    def local_fb(self, num_epoch):

        begin_time = time.time()

        for _ in range(num_epoch):
            self.model.train()
            for batch_idx, batch_data in enumerate(self.train_dataloader):
                (x, y, A) = batch_data
                if self.gpu:
                    x, y = x.to(self.device), y.to(self.device)
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
            loss = self.criterion(pred, y) + alpha * torch.sum(torch.abs(pred.reshape(-1) - a[:,1])) / self.batch_size
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
    
    def local_train(self):
        bytes_w = self.local_model_bytes

        begin_time = time.time()

        for _ in range(self.num_local_round):
            self.model.train()
            (x, y, a) = self.get_next_train_batch()
            if self.gpu:
                x, y = x.to(self.device), y.to(self.device)
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

    def model_eval(self, data):
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
                    x, y = x.to(self.device), y.to(self.device)
                
                pred = self.model(x)
                loss = self.criterion(pred, y)
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

        