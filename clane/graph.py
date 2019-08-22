import os
import pickle

import torch
from torch.utils.data import Dataset
from torch.nn.functional import pad

from loader import DatasetManager
from settings import DATA_PATH, PICKLE_PATH


torch.manual_seed(0)

class Graph(torch.utils.data.Dataset):
    def __init__(self, dataset, directed, dim, device):
        self.device = device
        self.node_ids, self.X, self.A, self.Y = DatasetManager().get(dataset, directed, device)
        self.Z = self.X.clone()
        if (dim is not None 
            and dim < self.X.shape[1]):
            from sklearn.decomposition import PCA

            pca = PCA(
                    n_components=dim,
                    random_state=0
                ) 
            self.X = pca.fit_transform(self.X.cpu())
            self.X = torch.FloatTensor(self.X).to(device)

        self.Z = self.X.clone()

    def __getitem__(self, index):
        return self.standard_Z[index], self.standard_Z[self.out_nbrs(index)], self.standard_Z[self.non_nbrs(index)]

    def __len__(self):
        return self.X.shape[0]
    
    def standardize(self):
        mean, std = self.Z.mean(), self.Z.std()
        self.standard_Z = self.Z.sub(mean).div(std)

    def out_nbrs(self, index):
        return torch.tensor(self.A[index], device=self.device, dtype=torch.long)
#        return (self.A[index]==1).nonzero().view(-1)
    
    def non_nbrs(self, index):
        return torch.tensor(
                       [i for i in range(len(self)) if i not in self.A[index]],
                       device=self.device,
                       dtype=torch.long
                   )

    @property
    def d(self):
        return self.Z.shape[-1]


def collate(samples):
    batch_z_src = list()
    batch_z_nbrs = list()
    batch_z_negs = list()
    max_nbrs, max_negs = 0, 0
    for z_src, z_nbrs, z_negs in samples:
        batch_z_src.append(z_src)
        batch_z_nbrs.append(z_nbrs)
        max_nbrs = max(max_nbrs, z_nbrs.shape[0])
        batch_z_negs.append(z_negs)
        max_negs = max(max_negs, z_negs.shape[0])

    batch_z_src = torch.stack(batch_z_src)
    dim = batch_z_src.shape[-1]
    batch_z_nbrs = [pad(z_nbrs, (0,0,0,max_nbrs-z_nbrs.shape[0]), value=-10) for z_nbrs in batch_z_nbrs]
    batch_z_negs = [pad(z_negs, (0,0,0,max_negs-z_negs.shape[0]), value=-10) for z_negs in batch_z_negs]
    batch_z_nbrs = torch.stack(batch_z_nbrs)
    batch_z_negs = torch.stack(batch_z_negs)
    nbr_mask = (batch_z_nbrs[:,:,0]!=-10).float()
    neg_mask = (batch_z_negs[:,:,0]!=-10).float()
    return batch_z_src, batch_z_nbrs, batch_z_negs, nbr_mask, neg_mask


if __name__ == '__main__':
    g = Graph(dataset = 'ppi', directed=True, dim=None, device=torch.device('cuda:2'))
    breakpoint()
