
import os
import numpy as np
from math import sqrt
from scipy import stats
from torch_geometric.data import InMemoryDataset
from sklearn.metrics import mean_squared_error
from torch_geometric import data as DATA
import torch
from ci import ci_fast

class AverageMeter(object):

    def __init__(self):
        self.sum = 0
        self.avg = 0
        self.val = 0
        self.count = 0
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count



def pearson(y, f):
    rp = np.corrcoef(y, f)[0, 1]
    return rp

def ci(y, f):
    return ci_fast(y, f)


def mse(y, f):
    return mean_squared_error(y, f)


def get_k(y_obs, y_pred):
    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)
    down = sum(y_pred * y_pred)
    if down == 0:
        return 0.0
    return sum(y_obs * y_pred) / float(down)


def r_squared_error(y_obs, y_pred):
    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)
    y_obs_mean = np.array([np.mean(y_obs) for _ in y_obs])
    y_pred_mean = np.array([np.mean(y_pred) for _ in y_pred])

    mult = sum((y_pred - y_pred_mean) * (y_obs - y_obs_mean))
    mult = mult * mult

    y_obs_sq = sum((y_obs - y_obs_mean) * (y_obs - y_obs_mean))
    y_pred_sq = sum((y_pred - y_pred_mean) * (y_pred - y_pred_mean))
    down = float(y_obs_sq * y_pred_sq)
    if down == 0:
        return 0.0
    return mult / down


def squared_error_zero(y_obs, y_pred):
    k = get_k(y_obs, y_pred)

    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)
    y_obs_mean = np.array([np.mean(y_obs) for _ in y_obs])
    upp = sum((y_obs - (k * y_pred)) * (y_obs - (k * y_pred)))
    down = sum((y_obs - y_obs_mean) * (y_obs - y_obs_mean))
    if down == 0:
        return 0.0
    return 1 - (upp / float(down))


def get_rm2(ys_orig, ys_line):
    r2 = r_squared_error(ys_orig, ys_line)
    r02 = squared_error_zero(ys_orig, ys_line)

    return r2 * (1 - np.sqrt(np.absolute((r2 * r2) - (r02 * r02))))

def top_overlap(y, f, top_k=10):
    top_k = min(top_k, len(y), len(f))

    top_pred_indices = np.argsort(f)[::-1][:top_k]


    top_true_indices = np.argsort(y)[::-1][:top_k]





    overlap = np.intersect1d(top_pred_indices, top_true_indices).size



    return overlap / top_k

def calculate_top_overlap(y, f):




    max_k = len(y)

    top1_overlap = top_overlap(y, f, top_k=1)
    top10_overlap = top_overlap(y, f, top_k=10)
    top_15_overlap = top_overlap(y, f, top_k=15)

    return {
        "top1_overlap": top_overlap(y, f, top_k=1),
        "top10_overlap": top_overlap(y, f, top_k=10),
        "top_15_overlap": top_overlap(y, f, top_k=15)
    }


def dcg_score(y, f, k=10):

    k = min(len(y), k)

    order = np.argsort(f)[::-1]
    gains = 2 ** y[order[:k]] - 1
    discounts = np.log2(np.arange(2, k + 2))


    discounts = np.maximum(discounts, 1e-10)
    return np.sum(gains / discounts)

def ndcg_score(y, f, k=10):
    actual_dcg = dcg_score(y, f, k)
    ideal_dcg = dcg_score(y, y, k)
    return actual_dcg / ideal_dcg if ideal_dcg > 0 else 0
