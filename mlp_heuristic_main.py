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
from utils.heuristic import CN, AA, RA
from models.MLP import MLPPolynomial

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
def train(model, optimizer, data, splits, device, A):
    model.train()
    optimizer.zero_grad()

    # Positive and negative edges for training
    pos_edge_index = splits['train']['pos_edge_label_index'].to(device)
    neg_edge_index = splits['train']['neg_edge_label_index'].to(device)

    # Labels for positive and negative edges (continuous regression labels)
    pos_edge_label = splits['train']['pos_edge_score'].to(device)
    neg_edge_label = splits['train']['neg_edge_score'].to(device)

    # Compute predictions for both positive and negative edges
    pos_pred = model(pos_edge_index[0], pos_edge_index[1])
    neg_pred = model(neg_edge_index[0], neg_edge_index[1])

    # Compute regression loss (MSE for continuous labels)
    pos_loss = F.mse_loss(pos_pred, pos_edge_label)
    neg_loss = F.mse_loss(neg_pred, neg_edge_label)
    loss = pos_loss + neg_loss
    loss.backward()

    # Optimizer step
    optimizer.step()
    visualize(pos_pred, pos_edge_label, save_path='./visualization_pos_train.png')
    visualize(neg_pred, neg_edge_label, save_path='./visualization_neg_train.png')

    return loss.item()



def valid(model, data, splits, device, A):
    model.eval()

    # Positive and negative edges for test
    pos_edge_index = splits['valid']['pos_edge_label_index'].to(device)
    neg_edge_index = splits['valid']['neg_edge_label_index'].to(device)

    # Labels for positive and negative edges (continuous regression labels)
    pos_edge_label = splits['valid']['pos_edge_score'].to(device)
    neg_edge_label = splits['valid']['neg_edge_score'].to(device)

    # Forward pass
    pos_pred = model(pos_edge_index[0], pos_edge_index[1])
    neg_pred = model(neg_edge_index[0], neg_edge_index[1])

    # Compute regression loss (MSE)
    pos_loss = F.mse_loss(pos_pred, pos_edge_label)
    neg_loss = F.mse_loss(neg_pred, neg_edge_label)
    loss = pos_loss + neg_loss

    return loss.item()
@torch.no_grad()
def test(model, data, splits, device, A):
    model.eval()

    # Positive and negative edges for test
    pos_edge_index = splits['test']['pos_edge_label_index'].to(device)
    neg_edge_index = splits['test']['neg_edge_label_index'].to(device)

    # Labels for positive and negative edges (continuous regression labels)
    pos_edge_label = splits['test']['pos_edge_score'].to(device)
    neg_edge_label = splits['test']['neg_edge_score'].to(device)

    # Forward pass
    pos_pred = model(pos_edge_index[0], pos_edge_index[1])
    neg_pred = model(neg_edge_index[0], neg_edge_index[1])
    visualize(pos_pred, pos_edge_label, save_path = './visualization_pos.png')
    visualize(neg_pred, neg_edge_label, save_path = './visualization_neg.png')

    # Compute regression loss (MSE)
    pos_loss = F.mse_loss(pos_pred, pos_edge_label)
    neg_loss = F.mse_loss(neg_pred, neg_edge_label)
    loss = pos_loss + neg_loss

    return loss.item()


def save_to_csv(file_path, model_name, heuristic, test_loss):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    file_exists = os.path.isfile(file_path)
    with open(file_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(['Model', 'Heuristic', 'Test_Loss'])
        writer.writerow([model_name, heuristic, test_loss])
def visualize(pred, true_label, save_path = './visualization.png'):

    pred = pred.cpu().detach().numpy()
    true_label = true_label.cpu().detach().numpy()
    plt.figure(figsize=(10, 6))
    plt.scatter(np.arange(len(true_label)), true_label, color='blue', label='True Score', alpha=0.6)
    plt.scatter(np.arange(len(pred)), pred, color='red', label='Prediction', alpha=0.6)

    plt.title('Predictions vs True Score')
    plt.xlabel('Sample Index')
    plt.ylabel('Value')
    plt.ylim(0, 1.5)
    plt.legend()

    plt.savefig(save_path)
    plt.close()

    print(f"Visualization saved at {save_path}")


def parseargs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=200, help="number of epochs")
    parser.add_argument('--dataset', type=str, default="Cora")
    parser.add_argument('--batch_size', type=int, default=512, help="batch size")
    parser.add_argument('--heuristic', type=str, default="CN")
    parser.add_argument('--model', type=str, default="MLP")
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
    if not args.use_feature:
        A_dense = A.toarray()
        A_tensor = torch.tensor(A_dense)
        data.x = A_tensor.float().to(device)

    model = MLPPolynomial(in_channels, hidden_channels=64, num_series=2, A=A).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    early_stopping = EarlyStopping(patience=10, verbose=True)
    for epoch in range(1, args.epochs + 1):
        start = time.time()
        loss = train(model, optimizer, data, splits, device, A)
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')
        val_loss = valid(model, data, splits, device, A)
        print(f'Validation Loss: {val_loss:.4f}')
        early_stopping(val_loss)
        if early_stopping.early_stop:
            print("Training stopped early!")
            break
    test_loss = test(model, data, splits, device, A)
    print(f'Test Result: Loss: {test_loss:.4f}')
    save_to_csv(f'./results/test_results_{args.dataset}.csv', args.model, args.heuristic, test_loss)
    print(f'Saved results.')

