import os

from torch.utils.data import Dataset, DataLoader
import torch

from settings import DATA_PATH


class CoraDataset(Dataset):
    def __init__(self):
        self.paper_ids = list()
        self.labels = list()
        self.c_x = None
        self.edges = list()
        with open(os.path.join(DATA_PATH, 'cora', 'cora.content'), 'r') as cora_content_io:
            while True:
                sample = cora_content_io.readline()
                if not sample: break
                paper_id, *content, label = sample.split('\t')
                self.paper_ids.append(int(paper_id))
                self.labels.append(label)
                content = torch.Tensor([[int(value) for value in content]])
                if self.c_x is not None:
                    self.c_x = torch.cat([self.c_x, content], 0)
                else:
                    self.c_x = content
        
        with open(os.path.join(DATA_PATH, 'cora', 'cora.cites'), 'r') as cora_edge_io:
            while True:
                sample = cora_edge_io.readline()
                if not sample: break
                cited, citing = sample.split('\t')
                self.edges.append((int(cited.strip()), int(citing.strip())))

    def __getitem__(self, index):
        return self.c_x[index],  self.paper_ids[index]

    def __len__(self):
        return len(self.paper_ids)

    def get_refs_of(self, index):
        src_idx = self.paper_ids[index]
        target_edges = (edge for edge in self.edges)
        return (self.paper_ids.index(edge[0]) \
                    if edge[0] == src_idx \
                    else self.paper_ids.index(edge[1]) \
                    for edge in target_edges)

class CoraDataLoader(DataLoader):
    pass


if __name__ == "__main__":
    cora = CoraDataset()
    for ref_idx in cora.get_refs_of(3):
        print(ref_idx)

