import os
import pickle

import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset

from settings import DATA_PATH, PICKLE_PATH

torch.manual_seed(0)
class GraphDataset(Dataset):
    def __init__(self,dataset, device, sampled, load):
        X = pickle.load(open(os.path.join(PICKLE_PATH, 'embedding', load),'rb'))
        self.X = torch.tensor(X, device=device)
        self.Z = self.X.clone()
        Y = pickle.load(open(os.path.join(DATA_PATH, dataset, f'{dataset}.labels'),'rb'))
        self.Y = torch.tensor(Y, device=device)
        self.A = torch.zeros(self.X.shape[0], self.X.shape[0], device=device)
        with open(os.path.join(DATA_PATH, dataset, f'{dataset}.edgelist'), 'r') as edge_io:
            while True:
                line = edge_io.readline()
                if not line: break
                src, dst = line.split(' ')
                self.A[int(src), int(dst)] = 1

        
        if sampled:
            sampled_link = open(
                    os.path.join(DATA_PATH, dataset, '{}_sampled.cites'.format(dataset))
                ).read().split('\n')[:-1]
            for link in sampled_link:
                dst, src = link.split(' ')
                self.A[self.id_list.index(src), self.id_list.index(dst)] = 0
                if (self.A.sum(0).add(self.A.sum(1))==-2).nonzero().sum()!=0:
                    self.A[self.id_list.index(src), self.id_list.index(dst)] = 1

    def __getitem__(self, index):
        return self.standard_Z[index], self.standard_Z[self.out_nbrs(index)], self.standard_Z[self.non_nbrs(index)]

    def __len__(self):
        return self.X.shape[0]
    
    def minmax_scale(self):
        max_, min_ = self.Z.max(), self.Z.min()
        self.scaled_Z = self.Z.sub(min_).div(max_-min_)
    
    def standard(self):
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

    def z(self, key):
        return self.G.nodes[key]['z']

    def save(self, dataset, pickle_name):
        import pickle, os
        from settings import PICKLE_PATH
        io = open(os.path.join(PICKLE_PATH, dataset, '{}'.format(pickle_name)), 'wb')
        pickle.dump(self.Z.cpu().data.numpy(), io)
        io.close()

    def load(self, config):
        import pickle, os
        from settings import PICKLE_PATH
        import torch
        io = open(os.path.join(PICKLE_PATH, config.dataset, config.model_tag), 'rb')
        loaded_Z = pickle.load(io)
        for idx, v in enumerate(self.G.nodes()):
            self.G.nodes()[v]['z'] = torch.tensor(loaded_Z[idx]).to(torch.device('cuda:{}'.format(config.gpu)))
        return self.node_list.index(key)
        io.close()
        self.G = nx.read_gpickle(os.path.join(PICKLE_PATH, 'network', config.model_tag))

class CoraDataset(GraphDataset):
    def __init__(self, device=torch.device('cpu'), sampled=False, load='cora_X'):
        super(CoraDataset, self).__init__('cora', device, sampled, load)

class CiteseerDataset(GraphDataset):
    def __init__(self, device=torch.device('cpu'), sampled=False, load='citeseer_X'):
        super(CiteseerDataset, self).__init__('citeseer', device, sampled, load)


if __name__ == "__main__":
    graph = CiteseerDataset()
    import pdb;    pdb.set_trace()
