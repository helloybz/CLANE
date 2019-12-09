import os

import torch
from torch.utils.data import Dataset

from clane.settings import DATA_PATH


class UndirectedGraphDataset(Dataset):
    def __init__(self):
        self.data.z = self.data.x.clone()

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

    @property
    def x(self):
        return self.data.x

    @property
    def edge_index(self):
        return self.data.edge_index

    @property
    def dim(self):
        return self.x.shape[1]


class CoraDataset(UndirectedGraphDataset):
    def __init__(self):
        from torch_geometric.datasets import Planetoid
        self.data = Planetoid(
            root=os.path.join(DATA_PATH, 'cora'),
            name='Cora'
        )[0]
        del self.data.train_mask
        del self.data.val_mask
        del self.data.test_mask
        super(CoraDataset, self).__init__()


class KarateDataset(UndirectedGraphDataset):
    def __init__(self):
        from torch_geometric.datasets import KarateClub
        self.data = KarateClub()[0]
        super(KarateDataset, self).__init__()
