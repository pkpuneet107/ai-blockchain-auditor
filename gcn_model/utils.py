import os
import json
import torch
from torch_geometric.data import Dataset, download_url, Data

class SmartContractDataset(Dataset):
    def __init__(self, root, file_path, transform=None, pre_transform=None):
        self.file_path = file_path
        super(SmartContractDataset, self).__init__(root, transform, pre_transform)
        
    @property
    def raw_file_names(self):
        return [os.path.basename(self.file_path)]
    
    @property
    def processed_file_names(self):
        return [f'data_{i}.pt' for i in range(self.len())]
    
    def download(self):
        # No download necessary as we use local files
        pass
    
    def len(self):
        with open(self.file_path, 'r') as f:
            data_list = json.load(f)
        return len(data_list)
    
    def process(self):
        with open(self.file_path, 'r') as f:
            data_list = json.load(f)
        
        for i, item in enumerate(data_list):
            # Extract graph structure
            edge_index = torch.tensor(item['graph'], dtype=torch.long)
            # Extract node features
            x = torch.tensor(item['node_features'], dtype=torch.float)
            # Extract target/label
            y = torch.tensor(int(item['targets']), dtype=torch.long)
            
            # Create PyTorch Geometric Data object
            data = Data(x=x, edge_index=edge_index, y=y)
            
            # Apply pre-processing if specified
            if self.pre_transform is not None:
                data = self.pre_transform(data)
            
            # Save to disk
            torch.save(data, os.path.join(self.processed_dir, f'data_{i}.pt'))
    
    def get(self, idx):
        data = torch.load(os.path.join(self.processed_dir, f'data_{idx}.pt'))
        return data