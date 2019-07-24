import os
import pickle

import torch
from torch.utils.data import Dataset

from preprocess import DatasetManager
from settings import DATA_PATH, PICKLE_PATH


torch.manual_seed(0)

class Graph(torch.utils.data.Dataset):
    def __init__(self, dataset, directed, dim, device):
        self.node_ids, self.X, self.A, self.Y = DatasetManager().get(dataset, device)
        self.Z = self.X.clone()
        if dim < self.X.shape[1]:
            try:
                self.X = torch.load(os.path.join(PICKLE_PATH, dataset, f'{dataset}_d{dim}'), map_location=device)
            except:
                from sklearn.decomposition import PCA

                pca = PCA(
                        n_components=dim,
                        random_state=0
                    ) 
                self.X = pca.fit_transform(self.X.cpu())
                self.X = torch.FloatTensor(self.X).to(device)
                torch.save(
                        self.X,
                        os.path.join(PICKLE_PATH, dataset, f'{dataset}_d{dim}'),
                    )

            self.Z = self.X.clone()

    def __getitem__(self, index):
        return self.standard_Z[index], self.standard_Z[self.out_nbrs(index)], self.standard_Z[self.non_nbrs(index)]

    def __len__(self):
        return self.X.shape[0]
    
    def minmax_scale(self):
        max_, min_ = self.Z.max(), self.Z.min()
        self.scaled_Z = self.Z.sub(min_).div(max_-min_)
    
    def standardize(self):
        mean, std = self.Z.mean(), self.Z.std()
        self.standard_Z = self.Z.sub(mean).div(std)

    def out_nbrs(self, index):
        return (self.A[index]==1).nonzero().view(-1)
    
    def in_nbrs(self, index):
        return (self.A.t()[index] == 1).nonzero().view(-1)
    
    def non_nbrs(self, index):
        return (self.A[index]==0).nonzero().view(-1)

    @property
    def d(self):
        return self.Z.shape[-1]


if __name__ == "__main__":
    g = Graph('cora', directed=True, device=torch.device('cuda'))
    breakpoint()

