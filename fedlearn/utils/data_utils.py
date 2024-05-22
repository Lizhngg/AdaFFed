import copy
import torch
import random
import json
import numpy as np
import pandas as pd
import os
from PIL import Image
from torchvision import datasets, transforms
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

import numpy as np
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
from torchvision.transforms import transforms
from pathlib import Path
from PIL import Image
import multiprocessing
import threading
import concurrent.futures

from fedlearn.utils.sampling import *
# from data.celeba.metadata_to_json import celeba_generate

class Fair_Dataset(Dataset):
    def __init__(self, X, Y, A):
        self.X = X
        self.Y = Y
        self.A = A

    def __getitem__(self, index):
        X = self.X[index]
        Y = self.Y[index]
        A = self.A[index]
        return (X, Y, A)

    def __len__(self):
        return self.X.shape[0]
    
    def dim(self):
        return self.X.shape[1:]
    
def mkdir(*args: str) -> tuple:
    for path in args:
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
    return args

def get_data_info(train_data, test_data):
    train_num = sum(train_data['num_samples'])
    test_num = sum(test_data['num_samples'])
    Ylabel = []
    Alabel = []
    for cdata in train_data['user_data'].values():
        Ylabel.append(np.unique(cdata.Y))
        Alabel.append(np.unique(cdata.A))

    for cdata in test_data['user_data'].values():
        Ylabel.append(np.unique(cdata.Y))
        Alabel.append(np.unique(cdata.A))
    Ylabel = np.unique(np.hstack(Ylabel))
    Alabel = np.unique(np.hstack(Alabel))

    A_info = np.zeros(len(Alabel))

    for cdata in test_data['user_data'].values():
        for i in range(len(Alabel)):
            A_info[i] += sum(cdata.A == i) 

    for cdata in train_data['user_data'].values():
        for i in range(len(Alabel)):
            A_info[i] += sum(cdata.A == i) 

    A_num = len(Alabel)
    Y_num = len(Ylabel)

    return {'train_num':train_num, 'test_num':test_num, 'Ylabel':Ylabel, 'Alabel':Alabel, 'A_num':A_num, 'Y_num':Y_num, 
            'train_samples': train_data['num_samples'], 'test_samples': test_data['num_samples'],'client_samples':[train_data['num_samples'][i] + test_data['num_samples'][i] for i in range(len(train_data['num_samples']))], 'A_info':A_info}

def get_data(options):
    """ 
    Returns train and test datasets:
    """

    data_name = options['data'].lower()
    data_settings = options['data setting']
    data_settings.update({'num_users':options['num_users']})
    options.update(data_settings)

    if data_name == 'adult':
        if data_settings.get('natural', False):
            train_path = "data/adult/split_data/normal/train.json"
            test_path = "data/adult/split_data/normal/test.json"
            train_data, test_data = read_data(train_path, 'adult', data_settings['sensitive_attr']), read_data(test_path, 'adult', data_settings['sensitive_attr'])
        elif data_settings['dirichlet']:
            train_path = "data/adult/raw_data/train.csv"
            test_path = "data/adult/raw_data/test.csv"
            save_path = f"data/adult/split_data/num_users={data_settings['num_users']} sensitive_attr={data_settings['sensitive_attr']} dirichlet={data_settings['alpha']} by_sensitive={data_settings['by sensitive']}/"
            
            split_train_path = save_path + "train.json"
            split_test_path = save_path + "test.json"

            if os.path.exists(split_train_path) and os.path.exists(split_test_path) and not data_settings.get('generate',False):
                train_data, test_data = read_data(split_train_path), read_data(split_test_path) 
            else:
                mkdir(save_path)
                adult_process()
                df = pd.read_csv(train_path)
                X, Y = df.drop('salary', axis=1).to_numpy().astype(np.float32),  df['salary'].to_numpy().astype(np.float32)
                colname = df.drop('salary', axis=1).columns.tolist()
                if data_settings['sensitive_attr'] == 'sex-race':
                    X, A, Y = adult_get_sensitive_feature(X, colname, data_settings['sensitive_attr'], Y)
                else:
                    X, A = adult_get_sensitive_feature(X, colname, data_settings['sensitive_attr'])
                if data_settings.get('by sensitive', False):
                    partition, stats = dirichlet(X, A, data_settings['num_users'], data_settings['alpha'])
                    split_train = split(partition['data_indices'], X, Y, A)
                    df = pd.read_csv(test_path)
                    X, Y = df.drop('salary', axis=1).to_numpy().astype(np.float32),  df['salary'].to_numpy().astype(np.float32)
                    colname = df.drop('salary', axis=1).columns.tolist()
                    if data_settings['sensitive_attr'] == 'sex-race':
                        X, A, Y = adult_get_sensitive_feature(X, colname, data_settings['sensitive_attr'], Y)
                    else:
                        X, A = adult_get_sensitive_feature(X, colname, data_settings['sensitive_attr'])
                    data_indices = partition_test_data(partition['separation'], A)
                    split_test = split(data_indices, X, Y, A)

                    with open(split_train_path,'w') as outfile:
                        json.dump(split_train, outfile)
                    with open(split_test_path, 'w') as outfile:
                        json.dump(split_test, outfile)
                
                    train_data, test_data = read_data(split_train_path), read_data(split_test_path) 
                else:
                    partition, stats = dirichlet(X, Y, data_settings['num_users'], data_settings['alpha'])
                    split_train = split(partition['data_indices'], X, Y, A)
                    df = pd.read_csv(test_path)
                    X, Y = df.drop('salary', axis=1).to_numpy().astype(np.float32),  df['salary'].to_numpy().astype(np.float32)
                    colname = df.drop('salary', axis=1).columns.tolist()
                    if data_settings['sensitive_attr'] == 'sex-race':
                        X, A, Y = adult_get_sensitive_feature(X, colname, data_settings['sensitive_attr'], Y)
                    else:
                        X, A = adult_get_sensitive_feature(X, colname, data_settings['sensitive_attr'])
                    data_indices = partition_test_data(partition['separation'], Y)
                    split_test = split(data_indices, X, Y, A)

                    with open(split_train_path,'w') as outfile:
                        json.dump(split_train, outfile)
                    with open(split_test_path, 'w') as outfile:
                        json.dump(split_test, outfile)
                
                    train_data, test_data = read_data(split_train_path), read_data(split_test_path) 

    elif data_name == 'celeba':
        if data_settings.get('generate', False) and data_settings.get('natural', False):
            celeba_generate()
        if data_settings.get('natural', False):
            path = "data/celeba/split_data/natural/"
            split_train_path = path + "train.json"
            split_test_path  = path + "test.json"
            with open(path + "all_data.json", 'rb') as file:
                data_split = json.load(file)
            train_data = {'users': [], 'user_data':{}, 'num_samples':[]}
            test_data = {'users': [], 'user_data':{}, 'num_samples':[]}

            fraction = 0.7
            for c in data_split['user_data']:
                user_num = len(data_split['user_data'][c]['x'])
                sample_num = round(user_num * fraction)
                ids = random.sample(range(user_num), sample_num)
                train_data['user_data'][c] = {'x': [data_split['user_data'][c]['x'][id] for id in ids],
                                              'y': [data_split['user_data'][c]['y'][id] for id in ids],
                                              'A': [data_split['user_data'][c]['A'][id] for id in ids]}
                train_data['users'].append(c)
                train_data['num_samples'].append(len(ids))
                ids = list(set(range(user_num)) - set(ids))
                test_data['user_data'][c] =  {'x': [data_split['user_data'][c]['x'][id] for id in ids],
                                              'y': [data_split['user_data'][c]['y'][id] for id in ids],
                                              'A': [data_split['user_data'][c]['A'][id] for id in ids]}
                test_data['users'].append(c)
                test_data['num_samples'].append(len(ids))
            
            with open(split_train_path,'w') as outfile:
                json.dump(train_data, outfile)
            with open(split_test_path, 'w') as outfile:
                json.dump(test_data, outfile)

            train_data, test_data = read_data(split_train_path, 'celeba'), read_data(split_test_path, 'celeba')
        else:
            sample_num = 220000
            save_path = f"data/celeba/split_data/sample_num={sample_num} num_users={data_settings['num_users']} sensitive_attr={data_settings['sensitive_attr']} dirichlet={data_settings['alpha']} by_sensitive={data_settings['by sensitive']}/"
            split_train_path = save_path + "train.npy"
            split_test_path = save_path + "test.npy"
            if os.path.exists(split_train_path) and os.path.exists(split_test_path) and (not data_settings.get('generate',False)):
                split_train, split_test = np.load(split_train_path, allow_pickle=True), np.load(split_test_path, allow_pickle=True)
                train_data, test_data = get_unsaved_data(split_train.item()), get_unsaved_data(split_test.item())
                print('celeba data processed.')
            else:
                mkdir(save_path)
                if data_settings['sensitive_attr'] == 'sex':
                    sensitive_attr = 'Male'
                elif data_settings['sensitive_attr'] == 'age':
                    sensitive_attr = 'Young'
                elif data_settings['sensitive_attr'] == 'sex-race':
                    sensitive_attr = ['Male', 'Pale_Skin']
                elif data_settings['sensitive_attr'] == 'race':
                    sensitive_attr = 'Pale_Skin'
                X, Y, A = celeba_data_processing(sensitive_attr, sample_num)
                shuffled_indices = np.random.permutation(X.shape[0])
                fraction = 0.7
                train_num = round(sample_num * fraction)
                train_idx, test_idx = shuffled_indices[:train_num],shuffled_indices[train_num:]
                train_X, train_Y, train_A = X[train_idx], Y[train_idx], A[train_idx]
                test_X, test_Y, test_A = X[test_idx], Y[test_idx], A[test_idx]

                del X, Y, A

                if data_settings.get('by sensitive', False):
                    partition, stats = dirichlet(train_X, train_A, data_settings['num_users'], data_settings['alpha'])
                    split_train = celeba_split(partition['data_indices'], train_X, train_Y, train_A)
                    del train_X, train_Y, train_A
                    data_indices = partition_test_data(partition['separation'], test_A)
                    split_test = celeba_split(data_indices, test_X, test_Y, test_A)
                    del test_X, test_Y, test_A

                else:
                    partition, stats = dirichlet(train_X, train_Y, data_settings['num_users'], data_settings['alpha'])
                    split_train = celeba_split(partition['data_indices'], train_X, train_Y, train_A)
                    del train_X, train_Y, train_A
                    data_indices = partition_test_data(partition['separation'], test_Y)
                    split_test = celeba_split(data_indices, test_X, test_Y, test_A)
                    del test_X, test_Y, test_A
                
                np.save(split_train_path, split_train)
                np.save(split_test_path, split_test)

                split_train, split_test = np.load(split_train_path, allow_pickle=True), np.load(split_test_path, allow_pickle=True)
                train_data, test_data = get_unsaved_data(split_train.item()), get_unsaved_data(split_test.item())
                print('celeba data processed.')
        

    elif data_name == 'compas':
        save_path = f"data/compas/split_data/num_users={data_settings['num_users']} sensitive_attr={data_settings['sensitive_attr']} dirichlet={data_settings['alpha']} by_sensitive={data_settings['by sensitive']}/"
        split_train_path = save_path + "train.json"
        split_test_path = save_path + "test.json"
        print("use compas.")

        if os.path.exists(split_train_path) and os.path.exists(split_test_path) and (not data_settings.get('generate',False)):
            train_data, test_data = read_data(split_train_path), read_data(split_test_path) 
        else:
            mkdir(save_path)
            X_train, X_test, y_train, y_test, A_train, A_test = compas_1_data_processing(data_settings['sensitive_attr'])
            
            if data_settings.get('by sensitive', False):
                partition, stats = dirichlet(X_train, A_train, data_settings['num_users'], data_settings['alpha'])
                split_train = split(partition['data_indices'], X_train, y_train, A_train)

                data_indices = partition_test_data(partition['separation'], A_test)
                split_test = split(data_indices, X_test, y_test, A_test)

                with open(split_train_path,'w') as outfile:
                    json.dump(split_train, outfile)
                with open(split_test_path, 'w') as outfile:
                    json.dump(split_test, outfile)
            
                train_data, test_data = read_data(split_train_path), read_data(split_test_path) 

            else:
                partition, stats = dirichlet(X_train, y_train, data_settings['num_users'], data_settings['alpha'])
                split_train = split(partition['data_indices'], X_train, y_train, A_train)

                data_indices = partition_test_data(partition['separation'], y_test)
                split_test = split(data_indices,X_test, y_test, A_test)

                with open(split_train_path,'w') as outfile:
                    json.dump(split_train, outfile)
                with open(split_test_path, 'w') as outfile:
                    json.dump(split_test, outfile)
            
                train_data, test_data = read_data(split_train_path), read_data(split_test_path) 
            
    elif data_name == 'bank':
        train_path = "data/bank/raw_data/train.csv"
        test_path = "data/bank/raw_data/test.csv"
        save_path = f"data/bank/split_data/num_users={data_settings['num_users']} sensitive_attr={data_settings['sensitive_attr']} dirichlet={data_settings['alpha']} by_sensitive={data_settings['by sensitive']}/"
        split_train_path = save_path + "train.json"
        split_test_path = save_path + "test.json"
        if os.path.exists(split_train_path) and os.path.exists(split_test_path) and (not data_settings.get('generate',False)):
            train_data, test_data = read_data(split_train_path), read_data(split_test_path) 
        else:
            mkdir(save_path)
            df = pd.read_csv(train_path)
            X, Y = df.drop('y', axis=1).to_numpy().astype(np.float64),  df['y'].to_numpy().astype(np.float64)
            colname = df.drop('y', axis=1).columns.tolist()
            X, A = bank_get_sensitive_feature(X, colname, data_settings['sensitive_attr'])
            if data_settings.get('by sensitive', False):
                partition, stats = dirichlet(X, A, data_settings['num_users'], data_settings['alpha'])
                split_train = split(partition['data_indices'], X, Y, A)
                df = pd.read_csv(test_path)
                X, Y = df.drop('salary', axis=1).to_numpy().astype(np.float32),  df['salary'].to_numpy().astype(np.float32)
                colname = df.drop('salary', axis=1).columns.tolist()
                X, A = bank_get_sensitive_feature(X, colname, data_settings['sensitive_attr'])
                data_indices = partition_test_data(partition['separation'], A)
                split_test = split(data_indices, X, Y, A)

                with open(split_train_path,'w') as outfile:
                    json.dump(split_train, outfile)
                with open(split_test_path, 'w') as outfile:
                    json.dump(split_test, outfile)
            
                train_data, test_data = read_data(split_train_path), read_data(split_test_path) 
            else:
                partition, stats = dirichlet(X, Y, data_settings['num_users'], data_settings['alpha'])
                split_train = split(partition['data_indices'], X, Y, A)
                df = pd.read_csv(test_path)
                X, Y = df.drop('y', axis=1).to_numpy().astype(np.float32),  df['y'].to_numpy().astype(np.float32)
                colname = df.drop('y', axis=1).columns.tolist()
                X, A = bank_get_sensitive_feature(X, colname, data_settings['sensitive_attr'])
                data_indices = partition_test_data(partition['separation'], Y)
                split_test = split(data_indices, X, Y, A)

                with open(split_train_path,'w') as outfile:
                    json.dump(split_train, outfile)
                with open(split_test_path, 'w') as outfile:
                    json.dump(split_test, outfile)
            
                train_data, test_data = read_data(split_train_path), read_data(split_test_path) 
    
    else:
        raise ValueError('Not support dataset {}!'.format(data_name))
    
    options['num_shape'] = train_data['user_data'][list(train_data['user_data'])[0]].dim()
    data_info = get_data_info(train_data, test_data)
    options['data_info'] = data_info
    print(data_info)
    return train_data, test_data

def adult_process():
    # Adult
    sensitive_attributes = ['sex']
    categorical_attributes = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'native-country']
    continuous_attributes = ["age", "fnlwgt", "education-num", "capital-gain", "capital-loss", "hours-per-week"]
    features_to_keep = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status',
                'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss','hours-per-week', 
                'native-country', 'salary']
    label_name = 'salary'

    adult = process_adult_csv('data/adult/raw_data/adult.data', label_name, ' >50K', sensitive_attributes, [' Female'], categorical_attributes, continuous_attributes, features_to_keep, na_values = [], header = None, columns = features_to_keep)
    test = process_adult_csv('data/adult/raw_data/adult.test', label_name, ' >50K.', sensitive_attributes, [' Female'], categorical_attributes, continuous_attributes, features_to_keep, na_values = [], header = None, columns = features_to_keep) # the distribution is very different from training distribution
    test['native-country_ Holand-Netherlands'] = 0
    test = test[adult.columns]

    adult_num_features = len(adult.columns)-1

    adult.to_csv('data/adult/raw_data/train.csv', index=None)
    test.to_csv('data/adult/raw_data/test.csv', index=None)
    
def process_adult_csv(filename, label_name, favorable_class, sensitive_attributes, privileged_classes, categorical_attributes, continuous_attributes, features_to_keep, na_values = [], header = 'infer', columns = None):
    """
    from https://github.com/yzeng58/Improving-Fairness-via-Federated-Learning/blob/main/FedFB/DP_load_dataset.py
    process the adult file: scale, one-hot encode
    only support binary sensitive attributes -> [gender, race] -> 4 sensitive groups 
    """
    skiprows = 1 if filename.endswith('test') else 0
    df = pd.read_csv(os.path.join(filename), delimiter = ',', header = header, na_values = na_values, skiprows=skiprows)
    if header == None: df.columns = columns
    df = df[features_to_keep]

    # apply one-hot encoding to convert the categorical attributes into vectors
    df = pd.get_dummies(df, columns = categorical_attributes)

    # normalize numerical attributes to the range within [0, 1]
    def scale(vec):
        minimum = min(vec)
        maximum = max(vec)
        return (vec-minimum)/(maximum-minimum)
    
    df[continuous_attributes] = df[continuous_attributes].apply(scale, axis = 0)
    df.loc[df[label_name] != favorable_class, label_name] = 0
    df.loc[df[label_name] == favorable_class, label_name] = 1
    df[label_name] = df[label_name].astype('category').cat.codes
    df['sex'] = df['sex'].map({' Male':0, ' Female':1}).astype('category')
    return df


def compas_1_data_processing(sensitive='sex-race'):
    #@title Load COMPAS dataset

    LABEL_COLUMN = 'two_year_recid'
    if sensitive == 'sex-race':
        sensitive_attributes = ['sex_Female', 'race_African-American']
    elif sensitive == 'race':
        sensitive_attributes = ['race_African-American']


    def get_data():
        data_path = "data/compas/raw_data/compas-scores-two-years.csv"
        df = pd.read_csv(data_path)
        FEATURES = [
            'age', 'c_charge_degree', 'race', 'age_cat', 'score_text', 'sex',
            'priors_count', 'days_b_screening_arrest', 'decile_score', 'is_recid',
            'two_year_recid'
        ]
        df = df[FEATURES]
        df = df[df.days_b_screening_arrest <= 30]
        df = df[df.days_b_screening_arrest >= -30]
        df = df[df.is_recid != -1]
        df = df[df.c_charge_degree != 'O']
        df = df[df.score_text != 'N/A']
        continuous_features = [
            'priors_count', 'days_b_screening_arrest', 'is_recid', 'two_year_recid'
        ]
        continuous_to_categorical_features = ['age', 'decile_score', 'priors_count']
        categorical_features = ['c_charge_degree', 'race', 'score_text', 'sex']
        # continuous_to_categorical_features = [ 'priors_count']
        # categorical_features = ['c_charge_degree', 'race', 'sex']

        # Functions for preprocessing categorical and continuous columns.
        def binarize_categorical_columns(input_df, categorical_columns=[]):
            # Binarize categorical columns.
            binarized_df = pd.get_dummies(input_df, columns=categorical_columns)
            return binarized_df

        def bucketize_continuous_column(input_df, continuous_column_name, bins=None):
            input_df[continuous_column_name] = pd.cut(
                input_df[continuous_column_name], bins, labels=False)

        for c in continuous_to_categorical_features:
            b = [0] + list(np.percentile(df[c], [20, 40, 60, 80, 90, 100]))
            if c == 'priors_count':
                b = list(np.percentile(df[c], [0, 50, 70, 80, 90, 100]))
            bucketize_continuous_column(df, c, bins=b)

        # df = binarize_categorical_columns(
        #     df,
        #     categorical_columns=categorical_features)

        df = binarize_categorical_columns(
            df,
            categorical_columns=categorical_features +
            continuous_to_categorical_features)

        to_fill = [
            u'decile_score_0', u'decile_score_1', u'decile_score_2',
            u'decile_score_3', u'decile_score_4', u'decile_score_5'
        ]
        for i in range(len(to_fill) - 1):
            df[to_fill[i]] = df[to_fill[i:]].max(axis=1)
            
        to_fill = [
            u'priors_count_0.0', u'priors_count_1.0', u'priors_count_2.0',
            u'priors_count_3.0', u'priors_count_4.0'
        ]
        for i in range(len(to_fill) - 1):
            df[to_fill[i]] = df[to_fill[i:]].max(axis=1)

        print(df.columns)
        features = [
            u'days_b_screening_arrest', u'c_charge_degree_F', u'c_charge_degree_M',
            u'race_African-American', u'race_Asian', u'race_Caucasian',
            u'race_Hispanic', u'race_Native American', u'race_Other',
            u'score_text_High', u'score_text_Low', u'score_text_Medium',
            u'sex_Female', u'sex_Male', u'age_0', u'age_1', u'age_2', u'age_3',
            u'age_4', u'age_5', u'decile_score_0', u'decile_score_1',
            u'decile_score_2', u'decile_score_3', u'decile_score_4',
            u'decile_score_5', u'priors_count_0.0', u'priors_count_1.0',
            u'priors_count_2.0', u'priors_count_3.0', u'priors_count_4.0'
        ]

        # # new
        # features = [
        #     u'days_b_screening_arrest', u'c_charge_degree_F', u'c_charge_degree_M',
        #     u'race_African-American', u'race_Asian', u'race_Caucasian',
        #     u'race_Hispanic', u'race_Native American', u'race_Other',
        #     u'sex_Female', u'sex_Male', u'age', u'priors_count_0.0', u'priors_count_1.0',
        #     u'priors_count_2.0', u'priors_count_3.0', u'priors_count_4.0'
        # ]
        # print(len(features))

        label = ['two_year_recid']

        df = df[features + label]
        return df, features, label

    df, feature_names, label_column = get_data()

    # if sensitive == 'race':
    #     df_w = df[df['race_Caucasian'] == 1]
    #     df_b = df[df['race_African-American'] == 1]
    #     df = pd.concat([df_w, df_b])

    from sklearn.utils import shuffle
    df = shuffle(df)
    N = len(df)
    train_df = df[:int(N * 0.66)]
    test_df = df[int(N * 0.66):]

    X_train_compas = np.array(train_df[feature_names])
    y_train_compas = np.array(train_df[label_column]).flatten()
    X_test_compas = np.array(test_df[feature_names])
    y_test_compas = np.array(test_df[label_column]).flatten()

    if sensitive == 'sex-race':

        # 0: male non-black, 1: female non-black, 2: male black, 3: female black
        A_train_compas = np.array(train_df[sensitive_attributes[0]] + train_df[sensitive_attributes[1]] * 2).flatten()
        A_test_compas = np.array(test_df[sensitive_attributes[0]] + test_df[sensitive_attributes[1]] * 2).flatten()

        sex_race_idx = [i for i, value in enumerate(feature_names) if (value.startswith('race') or value.startswith('sex')) ==True]
        X_train_compas = np.delete(X_train_compas, sex_race_idx, axis=1)
        X_test_compas = np.delete(X_test_compas, sex_race_idx, axis=1)

        print(X_train_compas.shape)
    
    elif sensitive == 'race':
        # 0: non-black, 1: black
        A_train_compas = np.array(train_df[sensitive_attributes]).flatten()
        A_test_compas = np.array(test_df[sensitive_attributes]).flatten()

        sen_idx = [i for i, value in enumerate(feature_names) if value.startswith('race')==True]
        X_train_compas = np.delete(X_train_compas, sen_idx, axis=1)
        X_test_compas = np.delete(X_test_compas, sen_idx, axis=1)

    print("compas process end.")

    return X_train_compas, X_test_compas, y_train_compas, y_test_compas, A_train_compas, A_test_compas


def celeba_data_processing(sensitive_attr, sample_num):
    f_identities = open(os.path.join('data', 'celeba', 'raw_data', 'identity_CelebA.txt'), 'r')
    identities = f_identities.read().split('\n')

    f_attributes = open(os.path.join('data', 'celeba', 'raw_data', 'list_attr_celeba.txt'), 'r')
    attributes = f_attributes.read().split('\n')

    tar = 'Smiling'

    sen_attr = sensitive_attr

    target_idx = attributes[1].split().index(tar)
    if type(sen_attr) == list:
        assert len(sen_attr) == 2
        sen_idx = [attributes[1].split().index(sen) for sen in sen_attr]
    elif type(sen_attr) == str:
        sen_idx = attributes[1].split().index(sen_attr)

    image = {}

    for line in attributes[2:]:
        info = line.split()
        if len(info) == 0:
            continue
        image_id = info[0]
        tar_img = (int(info[1:][target_idx]) + 1) / 2
        if type(sen_attr) == list:
            # 0: non-white female, 1: non-white male, 2: white female, 3:white male
            sen_img1 = (int(info[1:][sen_idx[0]]) + 1) / 2
            sen_img2 = (int(info[1:][sen_idx[1]]) + 1) / 2
            sen_img = sen_img1 + 2 * sen_img2
        elif type(sen_attr) == str:
            sen_img = (int(info[1:][sen_idx]) + 1) / 2

        image[image_id] = tar_img, sen_img

    images_path = Path(os.path.join('data', 'celeba', 'raw_data', 'img_align_celeba'))

    images_list = list(images_path.glob('*.jpg')) # list(images_path.glob('*.png'))
    images_list_str = [ str(x) for x in images_list ]
    images_ids = random.sample(images_list_str, sample_num)

    sample_target = []
    sample_sensitive = []
    for path in images_ids:
        sample_target.append(image[path[-10:]][0])
        sample_sensitive.append(image[path[-10:]][1])

    transform = transforms.Compose([
            transforms.CenterCrop((178, 178)), 
            transforms.Resize((128, 128)), 
            transforms.ToTensor(),
            # transforms.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5]) 
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
        ])
    
    print('start.')

    mp_img_loader = multiprocess_img_load(images_ids, transform)
    X = mp_img_loader.get_imgs().astype(np.float32)
    print(X.shape)
    print(type(X))

    # X1 = np.vstack(np.expand_dims([transform(Image.open(img)).numpy() for img in images_ids],axis=0))
    # print(np.sum(X != X1))
    print('end.')
    Y, A = np.array(sample_target), np.array(sample_sensitive)
    return X,Y,A

class multiprocess_img_load(object):
    def __init__(self, img_paths:list,transform, img_size=(3,128,128), n_thread=None) -> None:
        self.image_paths = img_paths
        self.img_size = img_size
        self.num_img = len(img_paths)
        self._mutex_put = threading.Lock()
        self.n_thread = n_thread if (n_thread is not None) else max(1, multiprocessing.cpu_count() - 2)
        self.transform = transform
    
    def get_imgs(self):
        self._buffer = np.zeros([self.num_img]+list(self.img_size))
        batch_size = round(self.num_img / self.n_thread)
        batch_idx = []
        for i in range(self.n_thread):
            idx = list(range(i * batch_size, (i+1) * batch_size)) if (i+1) * batch_size <= self.num_img else list(range(i * batch_size, self.num_img))
            batch_idx.append(idx)
        t_list = []
        for tid in range(self.n_thread):
            img_ids = list(range(tid * batch_size, (tid+1) * batch_size)) if (tid+1) * batch_size <= self.num_img else range(tid * batch_size, self.num_img)
            img_target = [self.image_paths[i] for i in img_ids]
            t = threading.Thread(target=self.load_image, args=(img_target, img_ids))
            t_list.append(t)
            t.start()

        for t in t_list:
            t.join()

        del t_list

        return self._buffer

    def load_image(self, img_names, img_ids):
        batch_images = np.vstack([np.expand_dims(self.transform(Image.open(img)).numpy(), axis=0) for img in img_names])
        self._mutex_put.acquire()
        self._buffer[img_ids] = batch_images
        self._mutex_put.release()


def celeba_split(data_indices, X ,Y ,A):
    split_data = {'users': [], 'user_data':{}, 'num_samples':[]}
    for i in range(len(data_indices)):
        split_data['users'].append(str(i))
        split_data['user_data'][str(i)] = {'x':X[data_indices[i],:],
                                      'y':Y[data_indices[i]],
                                      'A':A[data_indices[i]]}
        split_data['num_samples'].append(len(data_indices[i]))
    return split_data

def get_unsaved_data(data_split):
    for client in data_split['user_data']:
        X = np.array(data_split['user_data'][client]["x"]).astype(np.float32)
        Y = np.array(data_split['user_data'][client]["y"]).astype(np.float32).reshape(-1,1)
        A = np.array(data_split['user_data'][client]["A"]).astype(np.float32).reshape(-1,1)
        dataset = Fair_Dataset(X, Y, A)
        data_split['user_data'][client] = dataset
    return data_split

def bank_get_sensitive_feature(X, colname, sensitive_attr):
    if sensitive_attr == 'age':
        attr_idx = colname.index(sensitive_attr)
        A = X[:,attr_idx]
        X = np.delete(X, attr_idx, axis = 1)
    return X,A

def compas_get_sensitive_feature(X, colname, sensitive_attr):
    sex_attr = []
    race_attr = []
    for col in colname:
        if col.startswith('race'):
            race_attr.append(col)
        elif col.startswith('sex'):
            sex_attr.append(col)
    
    if sensitive_attr == 'sex':
        attr_idx = [colname.index(attr) for attr in sex_attr]
        A = np.argmax(X[:,attr_idx], axis =1 )  # [1: Male, 0: Female]
        X = np.delete(X, attr_idx, axis = 1)
    elif sensitive_attr == 'race':
        attr_idx = [colname.index(attr) for attr in race_attr]
        A = np.argmax(X[:,attr_idx], axis = 1) # ['African-American': 0,'Caucasian': 1,'Asian':2,'Hispanic':3]
        A[A>=1] = 1
        X = np.delete(X, attr_idx, axis = 1)
    elif sensitive_attr == 'non-sex':
        attr_idx = [colname.index(attr) for attr in sex_attr]
        A = np.argmax(X[:,attr_idx], axi = 1) 
    elif sensitive_attr == 'non-race':
        attr_idx = [colname.index(attr) for attr in race_attr] 
        A = np.argmax(X[:,attr_idx], axis = 1)
    return X, A

def split_celeba_data(ids: list):
    path = 'data/celeba/raw_data/img_align_celeba/'
    imgs = np.concatenate([np.expand_dims(np.array(Image.open(path + id)).transpose(2,0,1), axis=0) for id in ids], axis=0)
    
    return imgs

def partition_test_data(separation, targets):
    label_num = len(set(targets))
    targets_numpy = np.array(targets, dtype=np.int32)
    data_indices = [[] for _ in range(len(separation[0]))]
    data_idx_for_each_label = [
        np.where(targets_numpy == i)[0] for i in range(label_num)
    ]
    for k in range(label_num):
        distrib_cumsum = (np.cumsum(separation[k]) * len(data_idx_for_each_label[k])).astype(int)[:-1]
        data_indices = [
            np.concatenate((idx_j, idx.tolist())).astype(np.int64)
            for idx_j, idx in zip(
                data_indices, np.split(data_idx_for_each_label[k], distrib_cumsum)
            )
        ]
    
    return data_indices

def split(data_indices, X ,Y ,A):
    split_data = {'users': [], 'user_data':{}, 'num_samples':[]}
    for i in range(len(data_indices)):
        split_data['users'].append(str(i))
        split_data['user_data'][str(i)] = {'x':X[data_indices[i],:].tolist(),
                                      'y':Y[data_indices[i]].tolist(),
                                      'A':A[data_indices[i]].tolist()}
        split_data['num_samples'].append(len(data_indices[i]))
    return split_data


def adult_get_sensitive_feature(X, colname, sensitive, Y=None):
    sex_attr = 'sex'
    race_attr = []
    for col in colname:
        if col.startswith('race'):
            race_attr.append(col)
    if sensitive == "race":
        attr = 'race_ White'
        attr_idx = colname.index(attr)
        A = np.array(X[:,attr_idx])  
        # print(np.unique(A))
        del_idx = [colname.index(attr) for attr in race_attr]
        X = np.delete(X, del_idx, axis = 1)
    elif sensitive == "sex":
        attr_idx = colname.index(sex_attr)
        A = X[:, attr_idx] # [1: female, 0: male]
        X = np.delete(X, attr_idx, axis = 1)
    elif sensitive == "none-race":
        attr_idx = [colname.index(attr) for attr in race_attr]
        A = np.argmax(X[:,attr_idx], axis =1 ) 
    elif sensitive == "none-sex":
        attr_idx = colname.index(sex_attr)
        A = X[:, attr_idx] # [1: female, 0: male]
    elif sensitive == "sex-race":
        race_idx = [colname.index(attr) for attr in race_attr] 
        race_unused = [colname.index(attr) for attr in ['race_ Amer-Indian-Eskimo', 'race_ Asian-Pac-Islander', 'race_ Other']] 
        Y = Y[np.sum(X[:,race_unused],axis=1) == 0]
        X = X[np.sum(X[:,race_unused],axis=1) == 0,:]
        sex_idx = colname.index(sex_attr)
        A = (np.argmax(X[:,race_idx], axis =1) + X[:,sex_idx]) - 2
        X = np.delete(X, race_idx + [sex_idx], axis = 1)
        return X,A,Y


    else:
        print("error sensitive attr")
        exit()
    
    return X, A


def read_data(path, name=None, sensitive_process=None):
    if sensitive_process:
        with open(path, 'rb') as file:
            data_split = json.load(file)
        for client in data_split['users']:
            X = np.array(data_split['user_data'][client]["x"]).astype(np.float32)

            Y = np.array(data_split['user_data'][client]["y"]).astype(np.float32).reshape(-1,1)

            if name == 'adult':
                X, A = adult_get_sensitive_feature(X, sensitive_process)

            dataset = Fair_Dataset(X, Y, A)
            data_split['user_data'][client] = dataset

        return data_split
    
    else:
        with open(path, 'rb') as file:
            data_split = json.load(file)
        for client in data_split['users']:
            if name == 'celeba':
                X = split_celeba_data(data_split['user_data'][client]["x"]).astype(np.float32)
            else:
                X = np.array(data_split['user_data'][client]["x"]).astype(np.float32)

            Y = np.array(data_split['user_data'][client]["y"]).astype(np.float32).reshape(-1,1)

            A = np.array(data_split['user_data'][client]["A"]).astype(np.float32).reshape(-1,1)
                
            dataset = Fair_Dataset(X, Y, A)
            data_split['user_data'][client] = dataset
        return data_split
    
