import os

import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset

from settings import DATA_PATH, PICKLE_PATH


class GraphDataset(Dataset):
    def __init__(self, dataset, sampled, device=torch.device('cpu'), **kwargs):
        content_list= list() 
        label_list = list()
        self.device= device 
        self.id_list= list()
        with open(os.path.join(DATA_PATH, dataset, '{}.content'.format(dataset)),
                  'r') as content_io:
            while True:
                line = content_io.readline()
                if not line: break
                id_, *content, label = line.split('\t')
                self.id_list.append(id_)
                content = torch.tensor([float(value) for value in content]).to(device)
                content_list.append(content)
                label_list.append(label)
        
        if 'deepwalk' in kwargs.keys() and kwargs['deepwalk']:
            import pickle
            deepwalk = pickle.load(open(os.path.join(PICKLE_PATH, dataset, '{}_deepwalk'.format(dataset)), 'rb'))
            self.X = torch.tensor(deepwalk).to(device)
        else:
            self.X = torch.stack(content_list)

        self.Z = self.X.clone()
        label_set = list(set(label_list))
        self.Y = torch.tensor([label_set.index(label) for label in label_list]).to(device)
        del label_set, label_list, content_list
        self.A = torch.zeros(len(self.id_list), len(self.id_list)).to(device)
        with open(os.path.join(DATA_PATH, dataset,'{}.cites'.format(dataset)),
                  'r') as edge_io:
            while True:
                line = edge_io.readline()
                if not line: break
                target, source = line.split('\t')
                try:
                    target, source = self.id_list.index(target.strip()), self.id_list.index(source.strip())
                    self.A[source, target] = 1
                except ValueError:
                    pass
        self.A = self.A - torch.eye(len(self.id_list)).to(device)
        self.edges = (self.A==1).nonzero()

        self.S = torch.zeros(self.A.shape).to(device)

        self.batch_norm = torch.nn.BatchNorm1d(num_features=self.Z.shape[-1], affine=False).to(device)
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
    
    @property
    def normalized_Z(self):
        return self.batch_norm(self.Z) 

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
    def __init__(self, sampled=False, deepwalk=False, **kwargs):
        super(CoraDataset, self).__init__('cora', sampled, deepwalk=deepwalk, **kwargs)

class CiteseerDataset(GraphDataset):
    def __init__(self, sampled=False, **kwargs):
        super(CiteseerDataset, self).__init__('citeseer', sampled, **kwargs)


if __name__ == "__main__":
    graph = CiteseerDataset()
    import pdb;    pdb.set_trace()
