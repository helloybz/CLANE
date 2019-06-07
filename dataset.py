import os
import pickle

import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset

from settings import DATA_PATH, PICKLE_PATH


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
        return self.Z[index], self.Z[self.out_nbrs(index)], self.Z[self.non_nbrs(index)]

    def split(self, test_size):
        return train_test_split(list(range(len(self.id_list))), test_size=test_size)
        
    def __len__(self):
        return self.X.shape[0]

    def set_embedding(self, id_, embedding):
        idx = self.id_list.index(id_)
        self.Z[idx] = embedding.to(self.device)
    
    def out_nbrs(self, index):
        return (self.A[index] == 1).nonzero().squeeze(-1)
    
    def in_nbrs(self, index):
        return (self.A.t()[index] == 1).nonzero().squeeze(-1)
    
    def nbrs(self, index):
        return torch.cat([self.out_nbrs(index), self.in_nbrs(index)])

    def non_nbrs(self, index):
        return (self.A[index] == 0).nonzero().squeeze(-1)

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

class FlickrDataset(GraphDataset):
    def __init__(self, sampled=False, **kwargs):
        super(FlickrDataset, self).__init__('flickr', sampled, **kwargs)


if __name__ == "__main__":
    graph = CoraDataset(load='cora_deepwalk_d128')
    import pdb;    pdb.set_trace()
