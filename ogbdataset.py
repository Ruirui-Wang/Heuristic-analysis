import torch
from ogb.linkproppred import PygLinkPropPredDataset
from torch_geometric.transforms import RandomLinkSplit
from torch_sparse import SparseTensor
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import to_undirected, coalesce, remove_self_loops
import torch_geometric.transforms as T

# random split dataset
def randomsplit(dataset, val_ratio: float=0.10, test_ratio: float=0.2):
    data = dataset[0]
    data.edge_index, _ = coalesce(data.edge_index, None, num_nodes=data.num_nodes)
    data.edge_index, _ = remove_self_loops(data.edge_index)
    data.num_nodes = data.x.shape[0]
    transform = T.Compose([
        T.NormalizeFeatures(),
        RandomLinkSplit(is_undirected=True, num_val=val_ratio, num_test=test_ratio,add_negative_train_samples=True, split_labels=True)])
    train_data, val_data, test_data = transform(data)
    splits = {'train': {}, 'valid': {}, 'test': {}}
    splits['train'] = train_data
    splits['valid'] = val_data
    splits['test'] = test_data
    del data, train_data, val_data, test_data
    return splits

def loaddataset(name: str, use_valedges_as_input: bool, load=None):
    if name in ["Cora", "Citeseer", "Pubmed"]:
        dataset = Planetoid(root="dataset", name=name)
        splits = randomsplit(dataset)
        data = dataset[0]
        data.edge_index = to_undirected(splits["train"]["pos_edge_label_index"])
        edge_index = data.edge_index
        data.num_nodes = data.x.shape[0]
    else:
        dataset = PygLinkPropPredDataset(name=f'ogbl-{name}')
        splits = dataset.get_edge_split()
        data = dataset[0]
        edge_index = data.edge_index
    data.edge_weight = None
    data.adj_t = SparseTensor.from_edge_index(edge_index, sparse_sizes=(data.num_nodes, data.num_nodes))
    data.adj_t = data.adj_t.to_symmetric().coalesce()
    data.max_x = -1
    if name == "ppa":
        data.x = torch.argmax(data.x, dim=-1)
        data.max_x = torch.max(data.x).item()
    elif name == "ddi":
        data.x = torch.arange(data.num_nodes)
        data.max_x = data.num_nodes
    if load is not None:
        data.x = torch.load(load, map_location="cpu")
        data.max_x = -1
    data.full_adj_t = data.adj_t
    return data, splits

if __name__ == "__main__":
    loaddataset("Cora", False)
    loaddataset("Citeseer", False)
    loaddataset("Pubmed", False)
    loaddataset("ppa", False)
    loaddataset("collab", False)
    loaddataset("ddi", False)