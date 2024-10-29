import argparse
import csv
import os

import numpy as np
import scipy.sparse as ssp
import torch
import torch.nn.functional as F
import time

from matplotlib import pyplot as plt
from yacs.config import CfgNode

from utils.ogbdataset import loaddataset
from sklearn.metrics import roc_auc_score
from models.MLP import MLPPolynomial
from utils.heuristic import CN, AA, RA



def save_to_csv(file_path, heuristic, test_auc):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    file_exists = os.path.isfile(file_path)
    with open(file_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(['Model', 'Test_AUC'])
        writer.writerow([heuristic, test_auc])


def parseargs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=200, help="number of epochs")
    parser.add_argument('--dataset', type=str, default="Cora")
    parser.add_argument('--batch_size', type=int, default=512, help="batch size")
    parser.add_argument('--heuristic', type=str, default="CN")
    parser.add_argument('--use_feature', type=bool, default=False)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parseargs()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data, splits = loaddataset(args.dataset, True)
    data = data.to(device)
    in_channels = data.num_nodes
    edge_weight = torch.ones(data.edge_index.size(1), dtype=float)
    A = ssp.csr_matrix(
        (edge_weight, (data.edge_index[0].cpu(), data.edge_index[1].cpu())),
        shape=(data.num_nodes, data.num_nodes)
    )
    method_dict = {
        "CN": CN,
        "AA": AA,
        "RA": RA
    }
    for split in splits:
        pos_edge_score, _ = method_dict[args.heuristic](A, splits[split]['pos_edge_label_index'],
                                                        batch_size=args.batch_size)
        neg_edge_score, _ = method_dict[args.heuristic](A, splits[split]['neg_edge_label_index'],
                                                        batch_size=args.batch_size)
        splits[split]['pos_edge_score'] = torch.sigmoid(pos_edge_score)
        splits[split]['neg_edge_score'] = torch.sigmoid(neg_edge_score)

    all_preds = torch.cat([splits['test']['pos_edge_score'], splits['test']['neg_edge_score']], dim=0)
    all_labels = torch.cat([torch.ones(splits['test']['pos_edge_score'].size(0)), torch.zeros(splits['test']['neg_edge_score'].size(0))], dim=0).to(device)
    auc = roc_auc_score(all_labels, all_preds)
    print(f'Test AUC: {auc}')
    save_to_csv(f'./results/lp_results_{args.dataset}.csv', args.heuristic, auc)
    print(f'Saved results.')

