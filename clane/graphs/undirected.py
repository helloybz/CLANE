import os

import torch
from torch.utils.data import Dataset

from clane.settings import DATA_PATH


class CoraDataset(Dataset):
    def __init__(self):
        super(CoraDataset, self).__init__()
        from torch_geometric.datasets import Planetoid
        self.data = Planetoid(
            root=os.path.join(DATA_PATH, 'cora'),
            name='Cora'
        )[0]

        del self.data.train_mask
        del self.data.val_mask
        del self.data.test_mask

        self.data.z = self.data.x.clone()
        self.data.edge_weight = \
            self.data.edge_index.new_ones(self.data.num_edges)
        self.dim = self.data.x.shape[1]

    def __getitem__(self, index):
        # Compute src and dst from index.
        i, j = index // self.data.num_nodes, index % self.data.num_nodes

        edge = (self.data.edge_index[0] == i).mul(
            self.data.edge_index[1] == j
        ).any()
        return torch.tensor([i, j]), edge.float()

    def __len__(self):
        return self.data.x.shape[0]**2

    @property
    def z(self):
        return self.data.z
