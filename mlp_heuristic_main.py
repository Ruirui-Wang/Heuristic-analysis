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
from models.MLP import MLPPolynomialFeatures

class EarlyStopping:
    def __init__(self, patience=5, verbose=False):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False

    def __call__(self, val_loss):
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print("Early stopping triggered!")
def train(model, optimizer, data, splits, device, A, batch_size=512):
    model.train()
    optimizer.zero_grad()
    pos_edge_index = splits['train']['pos_edge_label_index'].to(device)
    neg_edge_index = splits['train']['neg_edge_label_index'].to(device)
    pos_edge_label = splits['train']['pos_edge_label'].to(device)
    neg_edge_label = splits['train']['neg_edge_label'].to(device)

    total_loss = 0
    pos_preds, neg_preds = [], []

    for i in range(0, pos_edge_index.size(1), batch_size):
        batch_pos_edge_index = pos_edge_index[:, i:i+batch_size]
        batch_pos_pred = model(batch_pos_edge_index[0], batch_pos_edge_index[1],
                               data.x[batch_pos_edge_index[0]], data.x[batch_pos_edge_index[1]])
        pos_preds.append(batch_pos_pred)
        batch_pos_loss = -torch.log(batch_pos_pred + 1e-15).mean()
        total_loss += batch_pos_loss

    for i in range(0, neg_edge_index.size(1), batch_size):
        batch_neg_edge_index = neg_edge_index[:, i:i+batch_size]
        batch_neg_pred = model(batch_neg_edge_index[0], batch_neg_edge_index[1],
                               data.x[batch_neg_edge_index[0]], data.x[batch_neg_edge_index[1]])
        neg_preds.append(batch_neg_pred)
        batch_neg_loss = -torch.log(1 - batch_neg_pred + 1e-15).mean()
        total_loss += batch_neg_loss

    total_loss.backward()
    optimizer.step()

    pos_pred = torch.cat(pos_preds, dim=0)
    neg_pred = torch.cat(neg_preds, dim=0)
    all_preds = torch.cat([pos_pred, neg_pred], dim=0)
    all_labels = torch.cat([torch.ones(pos_pred.size(0)), torch.zeros(neg_pred.size(0))], dim=0).to(device)
    auc = roc_auc_score(all_labels.cpu().detach().numpy(), all_preds.cpu().detach().numpy())

    return auc



def valid(model, data, splits, device, A):
    model.eval()

    # Positive and negative edges for test
    pos_edge_index = splits['valid']['pos_edge_label_index'].to(device)
    neg_edge_index = splits['valid']['neg_edge_label_index'].to(device)

    # Labels for positive and negative edges (continuous regression labels)
    pos_edge_label = splits['valid']['pos_edge_label'].to(device)
    neg_edge_label = splits['valid']['neg_edge_label'].to(device)

    # Forward pass
    pos_pred = model(pos_edge_index[0], pos_edge_index[1], data.x[pos_edge_index[0], :], data.x[pos_edge_index[1], :])
    neg_pred = model(neg_edge_index[0], neg_edge_index[1], data.x[neg_edge_index[0], :], data.x[neg_edge_index[1], :])

    all_preds = torch.cat([pos_pred, neg_pred], dim=0)
    all_labels = torch.cat([torch.ones(pos_pred.size(0)), torch.zeros(neg_pred.size(0))], dim=0).to(device)

    # Compute AUC
    auc = roc_auc_score(all_labels.cpu().detach().numpy(), all_preds.cpu().detach().numpy())

    return auc
@torch.no_grad()
def test(model, data, splits, device, A):
    model.eval()

    # Positive and negative edges for test
    pos_edge_index = splits['test']['pos_edge_label_index'].to(device)
    neg_edge_index = splits['test']['neg_edge_label_index'].to(device)

    # Labels for positive and negative edges (continuous regression labels)
    pos_edge_label = splits['test']['pos_edge_label'].to(device)
    neg_edge_label = splits['test']['neg_edge_label'].to(device)

    # Forward pass
    pos_pred = model(pos_edge_index[0], pos_edge_index[1], data.x[pos_edge_index[0],:], data.x[pos_edge_index[1],:])
    neg_pred = model(neg_edge_index[0], neg_edge_index[1], data.x[neg_edge_index[0],:], data.x[neg_edge_index[1],:])

    all_preds = torch.cat([pos_pred, neg_pred], dim=0)
    all_labels = torch.cat([torch.ones(pos_pred.size(0)), torch.zeros(neg_pred.size(0))], dim=0).to(device)

    # Compute AUC
    auc = roc_auc_score(all_labels.cpu().detach().numpy(), all_preds.cpu().detach().numpy())

    return auc


def save_to_csv(file_path, model_name, test_auc):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    file_exists = os.path.isfile(file_path)
    with open(file_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(['Model', 'Test_AUC'])
        writer.writerow([model_name+'_with_features', test_auc])
def visualize(pred, true_label, save_path = './visualization.png'):

    pred = pred.cpu().detach().numpy()
    true_label = true_label.cpu().detach().numpy()
    plt.figure(figsize=(10, 6))
    plt.scatter(np.arange(len(true_label)), true_label, color='blue', label='True label', alpha=0.6)
    plt.scatter(np.arange(len(pred)), pred, color='red', label='Prediction', alpha=0.6)

    plt.title('Predictions vs True label')
    plt.xlabel('Sample Index')
    plt.ylabel('Value')
    plt.ylim(0, 1.5)
    plt.legend()

    plt.savefig(save_path)
    plt.close()

    print(f"Visualization saved at {save_path}")


def parseargs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=500, help="number of epochs")
    parser.add_argument('--dataset', type=str, default="Citeseer")
    parser.add_argument('--batch_size', type=int, default=512, help="batch size")
    parser.add_argument('--model', type=str, default="MLP")
    parser.add_argument('--use_feature', type=bool, default=False)
    parser.add_argument('--early_stopping', type=bool, default=True)
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
    feature_size = data.x.size(1)

    model = MLPPolynomialFeatures(in_channels, feature_size, hidden_channels=64, num_series=2, A=A).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    early_stopping = EarlyStopping(patience=20, verbose=True)
    for epoch in range(1, args.epochs + 1):
        start = time.time()
        auc = train(model, optimizer, data, splits, device, A, batch_size=args.batch_size)
        print(f'Epoch: {epoch:03d}, AUC: {auc:.4f}')
        val_auc = valid(model, data, splits, device, A)
        print(f'Validation AUC: {val_auc:.4f}')
        test_auc = test(model, data, splits, device, A)
        print(f'Test AUC: {test_auc:.4f}')
        if args.early_stopping:
            early_stopping(val_auc)
            if early_stopping.early_stop:
                print("Training stopped early!")
                break
    test_auc = test(model, data, splits, device, A)
    print(f'Test Result: AUC: {test_auc:.4f}')
    save_to_csv(f'./results/lp_results_{args.dataset}.csv', args.model, test_auc)
    print(f'Saved results.')

