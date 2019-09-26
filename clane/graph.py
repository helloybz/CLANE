import torch
from torch.utils.data import Dataset

from loader import DatasetManager

torch.manual_seed(0)


class Graph(torch.utils.data.Dataset):
    def __init__(self, dataset, device=torch.device('cpu')):
        self.X, self.A, self.Y = DatasetManager().get(dataset, device)
        self.Z = self.X.clone()

    def __getitem__(self, index):
        return self.Z[index], self.Z[self.out_nbrs(index)], self.Z[self.non_nbrs(index)]
#        return self.standard_Z[index], self.standard_Z[self.out_nbrs(index)], self.standard_Z[self.non_nbrs(index)]

    def __len__(self):
        return self.X.shape[0]
   
    def clone_Z(self):
        self.clone_Z = self.Z.clone()

    def standardize(self):
        mean, std = self.Z.mean(), self.Z.std()
        self.standard_Z = self.Z.sub(mean).div(std)

    def out_nbrs(self, index):
        return torch.tensor(self.A[index], dtype=torch.long)
#        return (self.A[index]==1).nonzero().view(-1)
    
    def non_nbrs(self, index):
        # TODO: remove 기반 방법으로 변경
        return torch.tensor(
                       [i for i in range(len(self)) if i not in self.A[index]],
                       dtype=torch.long
                   )

    @property
    def dim(self):
        return self.Z.shape[-1]

