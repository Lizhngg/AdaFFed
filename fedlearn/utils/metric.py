import json
import numpy as np
import os
import time
import torch
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter
from torch.utils.data import Dataset
from PIL import Image


__all__ = ['mkdir', 'read_data', 'Metrics', 'MiniDataset', 'topk']


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    return path

class Metrics(object):
    def __init__(self, clients, options):
        self.options = options
        num_rounds = options['num_round'] + 1

        # Statistics in training procedure
        self.loss_on_train_data = [0] * num_rounds
        self.acc_on_train_data = [0] * num_rounds
        self.global_loss_on_train_data = [0] * num_rounds
        self.global_acc_on_train_data = [0] * num_rounds
        self.num_samples_on_train_data = [0] * num_rounds
        self.global_fairness_on_train_data = {f: [0] * num_rounds for f in options['fairness_measure']}
        self.global_fairness_on_train_data.update({f + ' disparity': [0] * num_rounds for f in options['fairness_measure']})

        # Statistics in test procedure
        self.loss_on_eval_data = [0] * num_rounds
        self.acc_on_eval_data = [0] * num_rounds
        self.global_loss_on_eval_data = [0] * num_rounds
        self.global_acc_on_eval_data = [0] * num_rounds
        self.num_samples_on_eval_data = [0] * num_rounds
        self.global_fairness_on_eval_data = [0] * num_rounds
        self.global_fairness_on_eval_data = {f: [0] * num_rounds for f in options['fairness_measure']}
        self.global_fairness_on_eval_data.update({f + ' disparity': [0] * num_rounds for f in options['fairness_measure']})


        self.result_path = mkdir(os.path.join('.\\result', self.options['data']))
        suffix = f"data={options['data']}_clients={options['num_users']}_alpha={options['data setting']['alpha']}_lr={options['local_lr']}"

        self.exp_name = '{}_{}_{}_{}'.format(time.strftime('%Y-%m-%d~%Hh%Mm%Ss'), options['algorithm'],
                                             options['model'], suffix)
        
        train_event_folder = mkdir(os.path.join(self.result_path, self.exp_name, 'train.event'))
        eval_event_folder = mkdir(os.path.join(self.result_path, self.exp_name, 'eval.event'))
        self.train_writer = SummaryWriter(train_event_folder)
        self.eval_writer = SummaryWriter(eval_event_folder)


    def update_model_stats(self, round_i, stats):
        # train stats
        self.global_loss_on_train_data[round_i] = stats['train_loss']
        self.global_acc_on_train_data[round_i] = stats['train_acc']
        for f in stats['train_fairness']:
            self.global_fairness_on_train_data[f][round_i] = stats['train_fairness'][f]
        self.train_writer.add_scalar('train_loss', self.global_loss_on_train_data[round_i], round_i)
        self.train_writer.add_scalar('train_acc', self.global_acc_on_train_data[round_i], round_i)
        for f in stats['train_fairness']:
            self.train_writer.add_scalar('train_' + f, self.global_fairness_on_train_data[f][round_i], round_i)

        # test stats
        self.global_loss_on_eval_data[round_i] = stats['test_loss']
        self.global_acc_on_eval_data[round_i] = stats['test_acc']
        for f in stats['test_fairness']:
            self.global_fairness_on_eval_data[f][round_i] = stats['test_fairness'][f]
        self.eval_writer.add_scalar('eval_loss', self.global_loss_on_eval_data[round_i], round_i)
        self.eval_writer.add_scalar('eval_acc', self.global_acc_on_eval_data[round_i], round_i)
        for f in stats['test_fairness']:
            self.eval_writer.add_scalar('eval_' + f, self.global_fairness_on_eval_data[f][round_i], round_i)


    def write(self):
        metrics = dict()

        # String
        metrics['data'] = self.options['data']
        metrics['num_round'] = self.options['num_round']
        metrics['eval_round'] = self.options['eval_round']
        metrics['local_lr'] = self.options['local_lr']
        metrics['num_round'] = self.options['num_round']
        metrics['batch_size'] = self.options['batch_size']
        # metrics['lambda'] = self.options['lambda']
        # metrics['q'] = self.options['q']
        # metrics['d'] = self.options['d']

        metrics['loss_on_train_data'] = self.loss_on_train_data
        metrics['acc_on_train_data'] = self.acc_on_train_data
        metrics['global_loss_on_train_data'] = self.global_loss_on_train_data
        metrics['global_acc_on_train_data'] = self.global_acc_on_train_data
        metrics['num_samples_on_train_data'] = self.num_samples_on_train_data
        for f in self.global_fairness_on_train_data:
            metrics['train_' + f] = self.global_fairness_on_train_data[f]
        

        metrics['loss_on_eval_data'] = self.loss_on_eval_data
        metrics['acc_on_eval_data'] = self.acc_on_eval_data
        metrics['global_loss_on_eval_data'] = self.global_loss_on_eval_data
        metrics['global_acc_on_eval_data'] = self.global_acc_on_eval_data
        metrics['num_samples_on_eval_data'] = self.num_samples_on_eval_data
        for f in self.global_fairness_on_eval_data:
            metrics['eval_' + f] = self.global_fairness_on_eval_data[f]

        metrics_dir = os.path.join(self.result_path, self.exp_name, 'metrics.json')

        with open(metrics_dir, 'w') as output_file:
            json.dump(str(metrics), output_file)

        summary_path, hyperparameter = get_summary_path(self.options)

        with open(summary_path + "\\test.txt", "a") as f:
            f.writelines(f"alpha: {self.options['data setting']['alpha']}")
            f.writelines("\n")
            f.writelines(f'hyperparameter: {hyperparameter}')
            f.writelines("\n")
            fairness_metric = {f: self.global_fairness_on_eval_data[f][-1] for f in self.global_fairness_on_eval_data}
            f.writelines(f'acc: {self.global_acc_on_eval_data[-1]}, fairness: {fairness_metric}')
            f.writelines("\n")
            f.writelines("\n")
    
def get_summary_path(options):
    if options['algorithm'] == 'wassffed':
        hyperparameter = f"OT_local_round={options['num_OT_local_round']},w_beta={options['w_beta']}"
    if options['algorithm'] == 'wasslocal':
        hyperparameter = f"OT_local_round={options['num_OT_local_round']},w_beta={options['w_beta']}"
    elif options['algorithm'] == 'fedfb':
        hyperparameter = f"local_round={options['num_local_round']},FB_alpha={options['FB_alpha']}"
    elif options['algorithm'] == 'afl':
        hyperparameter = f"weight_lr={options['weight_lr']}"
    elif options['algorithm'] == 'localfb':
        hyperparameter = f"FB_alpha={options['FB_alpha']}"
    elif options['algorithm'] == 'fedavg':
        hyperparameter = f"usual"
    elif options['algorithm'] == 'fairfed':
        hyperparameter = f"local_lr={options['local_lr']},fairfed_beta={options['fairfed_beta']}"
    elif options['algorithm'] == 'adaffed':
        hyperparameter = f"local_lr={options['local_lr'], options['a_value'], options['num_local_round'], options['eta']}"
    elif options['algorithm'] == 'localada':
        hyperparameter = f"local_lr={options['local_lr'], options['a_value'], options['num_local_round'], options['eta']}"
    summary_path = mkdir(os.path.join('.\\result', 'summary', options['data'], options['algorithm'])
                        )
    return summary_path, hyperparameter

