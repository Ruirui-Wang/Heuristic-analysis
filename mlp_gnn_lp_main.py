import argparse
import csv
import os

import numpy as np
import scipy.sparse as ssp
import torch
import torch.nn.functional as F
import time

from matplotlib import pyplot as plt
from sklearn.metrics import roc_auc_score
from yacs.config import CfgNode

from models.MLP import MLPPolynomialFeatures
from utils.ogbdataset import loaddataset
from utils.heuristic import CN, AA, RA
from models.GNN import GAT_Variant, GCN_Variant, SAGE_Variant, GIN_Variant, GAE_forall, InnerProduct, mlp_score

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

def create_GAE_model(cfg_model: CN,
                     in_channels,
                     model_name: str):
    if model_name in {'GAT', 'VGAE', 'GAE', 'GraphSage'}:
        raise NotImplementedError('Current model does not exist')
        # model = create_model(cfg_model)

    elif model_name == 'GAT_Variant':
        encoder = GAT_Variant(cfg_model.in_channels,
                              cfg_model.hidden_channels,
                              cfg_model.out_channels,
                              cfg_model.num_layers,
                              cfg_model.dropout,
                              cfg_model.heads,
                              )
    elif model_name == 'GCN_Variant':
        encoder = GCN_Variant(cfg_model.in_channels,
                              cfg_model.hidden_channels,
                              cfg_model.out_channels,
                              cfg_model.num_layers,
                              cfg_model.dropout,
                              )
    elif model_name == 'SAGE_Variant':
        encoder = SAGE_Variant(cfg_model.in_channels,
                               cfg_model.hidden_channels,
                               cfg_model.out_channels,
                               cfg_model.num_layers,
                               cfg_model.dropout,
                               )
    elif model_name == 'GIN_Variant':
        encoder = GIN_Variant(cfg_model.in_channels,
                              cfg_model.hidden_channels,
                              cfg_model.out_channels,
                              cfg_model.num_layers,
                              cfg_model.dropout,
                              cfg_model.mlp_layer
                              )
    decoder = MLPPolynomialFeatures(in_channels, cfg_model.out_channels, hidden_channels=64, num_series=2, A=A).to(device)


    return GAE_forall(encoder=encoder, decoder=decoder)

def train(model, optimizer, data, splits, device, batch_size=512):
    model.train()
    optimizer.zero_grad()

    pos_edge_index = splits['train']['pos_edge_label_index'].to(device)
    neg_edge_index = splits['train']['neg_edge_label_index'].to(device)
    pos_edge_label = splits['train']['pos_edge_label'].to(device)
    neg_edge_label = splits['train']['neg_edge_label'].to(device)

    z = model.encode(data.x, data.edge_index)

    pos_preds, neg_preds = [], []
    pos_labels, neg_labels = [], []

    for i in range(0, pos_edge_index.size(1), batch_size):
        pos_batch = pos_edge_index[:, i:i + batch_size]
        pos_label_batch = pos_edge_label[i:i + batch_size]
        pos_pred = model.decode(pos_batch[0], pos_batch[1], z[pos_batch[0]], z[pos_batch[1]])
        pos_preds.append(pos_pred)
        pos_labels.append(pos_label_batch)

    for i in range(0, neg_edge_index.size(1), batch_size):
        neg_batch = neg_edge_index[:, i:i + batch_size]
        neg_label_batch = neg_edge_label[i:i + batch_size]
        neg_pred = model.decode(neg_batch[0], neg_batch[1], z[neg_batch[0]], z[neg_batch[1]])
        neg_preds.append(neg_pred)
        neg_labels.append(neg_label_batch)

    pos_pred = torch.cat(pos_preds, dim=0)
    neg_pred = torch.cat(neg_preds, dim=0)
    pos_edge_label = torch.cat(pos_labels, dim=0)
    neg_edge_label = torch.cat(neg_labels, dim=0)

    pos_loss = F.mse_loss(pos_pred, pos_edge_label)
    neg_loss = F.mse_loss(neg_pred, neg_edge_label)
    loss = pos_loss + neg_loss
    loss.backward()
    optimizer.step()

    all_preds = torch.cat([pos_pred, neg_pred], dim=0)
    all_labels = torch.cat([torch.ones(pos_pred.size(0)), torch.zeros(neg_pred.size(0))], dim=0).to(device)
    auc = roc_auc_score(all_labels.cpu().detach().numpy(), all_preds.cpu().detach().numpy())

    visualize(pos_pred, pos_edge_label, save_path='./visualization_pos_train.png')
    visualize(neg_pred, neg_edge_label, save_path='./visualization_neg_train.png')

    return auc


@torch.no_grad()
def valid(model, data, splits, device, epoch):
    model.eval()

    # Positive and negative edges for validation
    pos_edge_index = splits['valid']['pos_edge_label_index'].to(device)
    neg_edge_index = splits['valid']['neg_edge_label_index'].to(device)

    # Labels for positive and negative edges (continuous regression labels)
    pos_edge_label = splits['valid']['pos_edge_label'].to(device)
    neg_edge_label = splits['valid']['neg_edge_label'].to(device)

    # Forward pass
    z = model.encode(data.x, data.edge_index)

    # Predict scores for both positive and negative edges
    pos_pred = model.decode(pos_edge_index[0], pos_edge_index[1], z[pos_edge_index[0]], z[pos_edge_index[1]])
    neg_pred = model.decode(neg_edge_index[0], neg_edge_index[1], z[pos_edge_index[0]], z[pos_edge_index[1]])

    # Compute regression loss (MSE)
    all_preds = torch.cat([pos_pred, neg_pred], dim=0)
    all_labels = torch.cat([torch.ones(pos_pred.size(0)), torch.zeros(neg_pred.size(0))], dim=0).to(device)

    # Compute AUC
    auc = roc_auc_score(all_labels.cpu().detach().numpy(), all_preds.cpu().detach().numpy())


    return auc


@torch.no_grad()
def test(model, data, splits, device):
    model.eval()

    # Positive and negative edges for test
    pos_edge_index = splits['test']['pos_edge_label_index'].to(device)
    neg_edge_index = splits['test']['neg_edge_label_index'].to(device)

    # Labels for positive and negative edges (continuous regression labels)
    pos_edge_label = splits['test']['pos_edge_label'].to(device)
    neg_edge_label = splits['test']['neg_edge_label'].to(device)

    # Forward pass
    z = model.encode(data.x, data.edge_index)

    # Predict scores for both positive and negative edges
    pos_pred = model.decode(pos_edge_index[0], pos_edge_index[1], z[pos_edge_index[0]], z[pos_edge_index[1]])
    neg_pred = model.decode(neg_edge_index[0], neg_edge_index[1], z[pos_edge_index[0]], z[pos_edge_index[1]])



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
        writer.writerow([model_name, test_auc])
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
    parser.add_argument('--dataset', type=str, default="Citeseer")
    parser.add_argument('--batch_size', type=int, default=512, help="batch size")
    parser.add_argument('--gnn', type=str, default="gcn")
    parser.add_argument('--model', type=str, default="GCN_Variant")
    parser.add_argument('--use_early_stopping', type=bool, default=True)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parseargs()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data, splits = loaddataset(args.dataset, True)
    data = data.to(device)
    with open('./yamls/cora/heart_gnn_models.yaml', "r") as f:
        cfg = CfgNode.load_cfg(f)
    cfg_model = eval(f'cfg.model.{args.model}')
    cfg_model.in_channels = splits['train']['x'].size(1)
    cfg_model.out_channels = 64
    cfg_score = eval(f'cfg.score.{args.model}')
    cfg.model.type = args.model
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

    model = create_GAE_model(cfg_model, data.num_nodes, args.model).to(device)
    encoder_lr = 0.01
    decoder_lr = 0.001
    optimizer = torch.optim.Adam([
        {'params': model.encoder.parameters(), 'lr': encoder_lr},
        {'params': model.decoder.parameters(), 'lr': decoder_lr}
    ])
    early_stopping = EarlyStopping(patience=20, verbose=True)
    for epoch in range(1, args.epochs + 1):
        start = time.time()
        auc = train(model, optimizer, data, splits, device, batch_size=args.batch_size)
        print(f'Epoch: {epoch:03d}, AUC: {auc:.4f}')
        val_auc = valid(model, data, splits, device, A)
        print(f'Validation AUC: {val_auc:.4f}')
        if args.use_early_stopping:
            early_stopping(val_auc)
            if early_stopping.early_stop:
                print("Training stopped early!")
                break
    test_auc = test(model, data, splits, device)
    print(f'Test Result: AUC: {test_auc:.4f}')
    save_to_csv(f'./results/lp_results_{args.dataset}.csv', args.model, test_auc)
    print(f'Saved results.')

