import os
from random import sample, shuffle

import torch
from torch.nn.functional import pad
from torch.utils.data import Dataset

from settings import DATA_PATH


class CoraDataset(Dataset):
    def __init__(self, **kwargs):
        self.paper_ids = list()
        self.labels = list()
        self.X = None
        self.A = None
        self.Z = None

        with open(os.path.join(DATA_PATH, 'cora', 'cora.content'),
                  'r') as cora_content_io:
            while True:
                sample = cora_content_io.readline()

                if not sample:
                    break

                paper_id, *content, label = sample.split('\t')
                self.paper_ids.append(int(paper_id))
                self.labels.append(label.strip())
                content = torch.Tensor([[int(value) for value in content]])

                if self.X is not None:
                    self.X = torch.cat([self.X, content], 0)
                else:
                    self.X = content

        self.A = torch.zeros(len(self.paper_ids), len(self.paper_ids))

        with open(os.path.join(DATA_PATH, 'cora', 'cora.cites'),
                  'r') as cora_edge_io:
            while True:
                sample = cora_edge_io.readline()
                if not sample:
                    break

                cited, citing = sample.split('\t')
                cited = self.paper_ids.index(int(cited))
                citing = self.paper_ids.index(int(citing))
                self.A[cited, citing] = 1

        self.X = self.X.to(kwargs['device'])
        self.A = self.A.to(kwargs['device'])
        self.Z = self.X.clone()
        label_categories = list(set(self.labels))
        self.labels = [label_categories.index(label) for label in self.labels]

    def __getitem__(self, index):
        return self.Z[index]

    def __len__(self):
        return len(self.paper_ids)

    def get_all_edges(self):
        for pair in self.A.nonzero():
            yield pair

    def get_all_non_edges(self):
        for z1, row in enumerate(self.A):
            yield z1, (row == 0).nonzero()

class CiteseerDataset:
    pass

if __name__ == "__main__":
    dataset = CoraDataset(device=torch.device('cuda'))
    print(dataset.get_all_edges(target_id=3))
