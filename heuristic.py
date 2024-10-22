from __future__ import division

import torch
import numpy as np
from tqdm import tqdm
from torch_geometric.data import DataLoader


def CN(A, edge_index, batch_size=100000):
    # Common Neighbor heuristic score.
    link_loader = DataLoader(range(edge_index.size(1)), batch_size)
    scores = []
    for ind in link_loader:
        src, dst = edge_index[0, ind], edge_index[1, ind]
        # Number of common neighbors is simply the dot product of the adjacency rows.
        cur_scores = np.array(A[src].multiply(A[dst]).sum(1)).flatten()
        scores.append(cur_scores)
    scores = np.concatenate(scores, 0)
    return torch.FloatTensor(scores), edge_index


def AA(A, edge_index, batch_size=100000):
    # The Adamic-Adar heuristic score.
    multiplier = 1 / np.log(A.sum(axis=0))
    multiplier[np.isinf(multiplier)] = 0
    A_ = A.multiply(multiplier).tocsr()
    link_loader = DataLoader(range(edge_index.size(1)), batch_size)
    scores = []
    # for ind in tqdm(link_loader):
    for ind in link_loader:
        src, dst = edge_index[0, ind], edge_index[1, ind]
        cur_scores = np.array(np.sum(A[src].multiply(A_[dst]), 1)).flatten()
        scores.append(cur_scores)
    scores = np.concatenate(scores, 0)
    return torch.FloatTensor(scores), edge_index


def RA(A, edge_index, batch_size=100000, beta=0.5, A2=None, gamma=0.1, num_nodes=0):
    # The Adamic-Adar heuristic score.
    # multiplier = 1 / np.log(A.sum(axis=0))
    multiplier = 1 / (np.power(A.sum(axis=0), beta))
    multiplier[np.isinf(multiplier)] = 0
    A_ = A.multiply(multiplier).tocsr()
    if A2 is not None:
        A2_ = A2.multiply(multiplier).tocsr()
    link_loader = DataLoader(range(edge_index.size(1)), batch_size)
    scores = []
    for ind in tqdm(link_loader):
        src, dst = edge_index[0, ind], edge_index[1, ind]
        cur_scores = np.array(np.sum(A[src].multiply(A_[dst]), 1)).flatten()
        # pdb.set_trace()
        if A2 is not None and gamma != 0:
            cur_scores2 = np.array(np.sum(A[src].multiply(A2_[dst]), 1)).flatten()
            cur_scores3 = np.array(np.sum(A[dst].multiply(A2_[src]), 1)).flatten()
            cur_scores4 = np.array(np.sum(A2[src].multiply(A2_[dst]), 1)).flatten()
            # cur_scores += 0.1 * (cur_scores2)
            cur_scores += gamma * (cur_scores2 + cur_scores3) + (gamma * gamma) * cur_scores4
        scores.append(cur_scores)
    scores = np.concatenate(scores, 0)
    return torch.FloatTensor(scores), edge_index

