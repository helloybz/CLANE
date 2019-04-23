import os

import numpy as np
import torch
from torch.utils.data import Dataset

from settings import DATA_PATH


class GraphDataset(Dataset):
    def __init__(self, dataset, sampled, device=torch.device('cpu'), **kwargs):
        content_list= list() 
        id_list= list()
        label_list = list()
        
        with open(os.path.join(DATA_PATH, dataset, '{}.content'.format(dataset)),
                  'r') as content_io:
            while True:
                line = content_io.readline()
                if not line: break
                id_, *content, label = line.split('\t')
               
                id_list.append(id_)
                content = torch.tensor([float(value) for value in content]).to(device)
                content_list.append(content)
                label_list.append(label)
        
        self.X = torch.stack(content_list)
        self.Z = self.X.clone()
        label_set = list(set(label_list))
        self.Y = [label_set.index(label) for label in label_list]
        del label_set, label_list, content_list
        self.A = torch.zeros(len(id_list), len(id_list)).to(device)
        with open(os.path.join(DATA_PATH, dataset,'{}.cites'.format(dataset)),
                  'r') as edge_io:
            while True:
                line = edge_io.readline()
                if not line: break
                target, source = line.split('\t')
                target, source = id_list.index(target.strip()), id_list.index(source.strip())
                self.A[target, source] = 1
        self.S = torch.zeros(self.A.shape).to(device)
        self.P = torch.zeros(self.A.shape).to(device)
        # TODO: re work sampled case
#        if sampled:
#            from random import sample
#            for node in self.G.nodes():
#                if self.G.in_degree(node) > 1:
#                    chosen_edge = sample(list(self.G.in_edges(node)), k=1)[0]
#                    self.G.remove_edge(*chosen_edge)
#                    if list(nx.isolates(self.G)):
#                        self.G.add_edge(*chosen_edge)
    
    def __getitem__(self, index):
        return self.Z[index]

    def __len__(self):
        return nx.number_of_nodes(self.G)

    def index(self, key):
        return self.node_list.index(key)
    def x(self, key):
        return self.G.nodes[key]['x']
    
    def z(self, key):
        return self.G.nodes[key]['z']
   
    @property
    def d(self):
        return self.Z.shape[-1]

    def z(self, key):
        return self.G.nodes[key]['z']

    def save(self, config):
        import pickle, os
        from settings import PICKLE_PATH
        io = open(os.path.join(PICKLE_PATH, config.dataset, config.model_tag), 'wb')
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
        io.close()
        self.G = nx.read_gpickle(os.path.join(PICKLE_PATH, 'network', config.model_tag))

class CoraDataset(GraphDataset):
    def __init__(self, sampled=False, **kwargs):
        super(CoraDataset, self).__init__('cora', sampled, **kwargs)

class CiteseerDataset(GraphDataset):
    def __init__(self, sampled=False, **kwargs):
        super(CiteseerDataset, self).__init__('citeseer', sampled, **kwargs)


if __name__ == "__main__":
    import torch
    import pdb

    dataset = CoraDataset(sampled=True, device=torch.device('cuda'))
    pdb.set_trace()
