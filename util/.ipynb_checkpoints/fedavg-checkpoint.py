# util/fedavg.py
import copy
import torch

def FedAvg(w, dict_len):
    """
    w: list of client weights
    dict_len: list of weights for each client (e.g. m_i * exp(-eta))
    """
    w_avg = copy.deepcopy(w[0])
    total_weight = sum(dict_len)
    
    for k in w_avg.keys():
        w_avg[k] = w_avg[k] * dict_len[0]
        for i in range(1, len(w)):
            w_avg[k] += w[i][k] * dict_len[i]
        w_avg[k] = w_avg[k] / total_weight
    return w_avg