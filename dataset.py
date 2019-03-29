import os

import networkx as nx
import numpy as np
import torch
from torch.utils.data import Dataset

from settings import DATA_PATH


class GraphDataset(Dataset):
    def __init__(self, dataset, sampled, **kwargs):
        self.G = nx.DiGraph()

        with open(os.path.join(DATA_PATH, dataset,'{}.cites'.format(dataset)),
                  'r') as edge_io:
            while True:
                line = edge_io.readline()
                if not line:
                    break

                target, source = line.split('\t')
                self.G.add_edge(source.strip(), target.strip())

        with open(os.path.join(DATA_PATH, dataset, '{}.content'.format(dataset)),
                  'r') as content_io:
            while True:
                line = content_io.readline()
                if not line:
                    break

                id_, *content, label = line.split('\t')
                self.G.nodes[id_]['x'] = torch.tensor([float(value) for value in content]).to(kwargs['device'])
                self.G.nodes[id_]['z'] = self.G.nodes[id_]['x'].clone().to(kwargs['device'])
                self.G.nodes[id_]['label'] = label.strip()

        if sampled:
            from random import sample
            for node in self.G.nodes():
                if self.G.in_degree(node) > 1:
                    chosen_edge = sample(list(self.G.in_edges(node)), k=1)[0]
                    self.G.remove_edge(*chosen_edge)
                    if list(nx.isolates(self.G)):
                        self.G.add_edge(*chosen_edge)
    
             
    def __getitem__(self, index):
        return self.Z[index]

    def __len__(self):
        return nx.number_of_nodes(self.G)

    def x(self, key):
        return self.G.nodes[key]['x']
    
    def z(self, key):
        return self.G.nodes[key]['z']
    @property
    def X(self):
        return torch.stack(list(nx.get_node_attributes(self.G, 'x').values()))

    @property
    def Z(self):
        return torch.stack(list(nx.get_node_attributes(self.G, 'z').values()))

    @property
    def Y(self):
        labels = nx.get_node_attributes(self.G, 'label').values() 
        label_set = list(set(labels))
        labels = [label_set.index(label) for label in labels]
        return torch.tensor(labels)
   
    @property
    def feature_size(self):
        return self.Z.shape[-1]

    def z(self, key):
        return self.G.nodes[key]['z']

    def save(self, config):
        import pickle, os
        from settings import PICKLE_PATH
        io = open(os.path.join(PICKLE_PATH, config.dataset, config.model_tag), 'wb')
        pickle.dump(self.Z.cpu().data.numpy(), io)
        io.close()
        nx.write_gpickle(self.G, os.path.join(PICKLE_PATH, 'network', config.model_tag))

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
