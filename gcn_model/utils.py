import os
import json
import torch
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader


class GraphDataset(Dataset):
    def __init__(self, json_path):
        super().__init__()
        with open(json_path, 'r') as f:
            self.graphs = json.load(f)

    def len(self):
        return len(self.graphs)

    def get(self, idx):
        sample = self.graphs[idx]
        x = torch.tensor(sample['node_features'], dtype=torch.float)
        edge_index = torch.tensor(sample['graph'], dtype=torch.long).t().contiguous()

        # Handle label (either from 'vuln_type' or fallback to 'targets')
        label = int(sample.get('vuln_type', sample['targets']))
        y = torch.tensor([label], dtype=torch.long)

        return Data(x=x, edge_index=edge_index, y=y)


def get_dataloaders(train_path, valid_path, batch_size=32, shuffle=True):
    train_dataset = GraphDataset(train_path)
    valid_dataset = GraphDataset(valid_path)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, valid_loader
