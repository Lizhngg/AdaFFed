
import numpy as np
from typing import Dict, List, Tuple
from collections import Counter

def dirichlet(X, targets, num_users: int, alpha: float, least_samples: int = 10) -> Tuple[List[List[int]], Dict]:
    """
    https://github.com/KarhouTam/FL-bench/blob/master/data/utils/schemes/dirichlet.py
    Dirichlet: Refers to Measuring the Effects of Non-Identical Data Distribution for Federated Visual Classification (FedAvgM). 
    Dataset would be splitted according to Dir(a). Smaller a means stronger label heterogeneity.

    --alpha or -a: The parameter for controlling intensity of label heterogeneity.
    --least_samples or -ls: The parameter for defining the minimum number of samples each client would be distributed. 
    A small --least_samples along with small --alpha or big --client_num might considerablely prolong the partition.
    """
    try:
        label_num = len(set(targets))
    except:
        label_num = len(np.unique(targets))
    min_size = 0
    stats = {}
    partition = {"separation": None, "data_indices": None}

    targets_numpy = np.array(targets, dtype=np.int32)
    data_idx_for_each_label = [
        np.where(targets_numpy == i)[0] for i in range(label_num)
    ]

    while min_size < least_samples:
        distrib_lst = []
        data_indices = [[] for _ in range(num_users)]
        for k in range(label_num):
            np.random.shuffle(data_idx_for_each_label[k])
            distrib = np.random.dirichlet(np.repeat(alpha, num_users))
            distrib = np.array(
                [
                    p * (len(idx_j) < len(targets_numpy) / num_users)
                    for p, idx_j in zip(distrib, data_indices)
                ]
            )
            distrib = distrib / distrib.sum()
            distrib_cumsum = (np.cumsum(distrib) * len(data_idx_for_each_label[k])).astype(int)[:-1]
            data_indices = [
                np.concatenate((idx_j, idx.tolist())).astype(np.int64)
                for idx_j, idx in zip(
                    data_indices, np.split(data_idx_for_each_label[k], distrib_cumsum)
                )
            ]
            distrib_lst.append(distrib)
            min_size = min([len(idx_j) for idx_j in data_indices])

    for i in range(num_users):
        stats[i] = {"x": None, "y": None}
        stats[i]["x"] = len(targets_numpy[data_indices[i]])
        stats[i]["y"] = Counter(targets_numpy[data_indices[i]].tolist())

    num_samples = np.array(list(map(lambda stat_i: stat_i["x"], stats.values())))
    stats["sample per client"] = {
        "std": num_samples.mean(),
        "stddev": num_samples.std(),
    }

    partition["data_indices"] = data_indices
    partition["separation"]   = distrib_lst
    return partition, stats

