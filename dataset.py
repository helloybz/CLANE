import os

import networkx as nx
import numpy as np
import torch
from torch.utils.data import Dataset

from settings import DATA_PATH


class GraphDataset(Dataset):
    def __init__(self):
        self.G = nx.DiGraph()

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

    def load(self, config):
        import pickle, os
        from settings import PICKLE_PATH
        import torch
        io = open(os.path.join(PICKLE_PATH, config.dataset, config.model_tag), 'rb')
        loaded_Z = pickle.load(io)
        for idx, v in enumerate(self.G.nodes()):
            self.G.nodes()[v]['z'] = torch.tensor(loaded_Z[idx]).to(torch.device('cuda:{}'.format(config.gpu)))

class CoraDataset(GraphDataset):
    def __init__(self, **kwargs):
        super(CoraDataset, self).__init__()

        with open(os.path.join(DATA_PATH, 'cora', 'cora.cites'),
                  'r') as cora_edge_io:
            while True:
                sample = cora_edge_io.readline()
                if not sample:
                    break

                cited, citing = sample.split('\t')
                self.G.add_edge(citing.strip(), cited.strip())

        with open(os.path.join(DATA_PATH, 'cora', 'cora.content'),
                  'r') as cora_content_io:
            while True:
                sample = cora_content_io.readline()
                if not sample:
                    break

                paper_id, *content, label = sample.split('\t')
                self.G.nodes[paper_id]['x'] = torch.tensor([float(value) for value in content]).to(kwargs['device'])
                self.G.nodes[paper_id]['z'] = self.G.nodes[paper_id]['x'].clone().to(kwargs['device'])
                self.G.nodes[paper_id]['label'] = label.strip()


class CiteseerDataset(GraphDataset):
    def __init__(self, **kwargs):
        super(CiteseerDataset, self).__init__(**kwargs)

        with open(os.path.join(DATA_PATH, 'citeseer', 'citeseer.cites'),
                  'r') as citeseer_edge_io:
            while True:
                sample = citeseer_edge_io.readline()
                if not sample:
                    break

                cited, citing = sample.split('\t')
                self.G.add_edge(citing.strip(), cited.strip())

        with open(os.path.join(DATA_PATH, 'citeseer', 'citeseer.content'),
                  'r') as citseer_content_io:
            while True:
                sample = citseer_content_io.readline()
                if not sample:
                    break

                paper_id, *content, label = sample.split('\t')
                self.G.nodes[paper_id]['x'] = np.array([float(value) for value in content])
                self.G.nodes[paper_id]['label'] = label.strip()

if __name__ == "__main__":
    dataset = CoraDataset()
