import logging
import os

import torch

from .graph import InMemoryDataset
from clane import g


class Cora(InMemoryDataset):
    def __init__(self):
        self.x = torch.empty([0,])
        node_ids = list()
        self.y = list()

        content_io = open(os.path.join(g.paths["data"], "cora", "cora.content"))
        while True:
            line = content_io.readline()
            if not line: break
            node_id, *values, label = line.split()
            node_ids.append(node_id)
            values = torch.tensor([float(value) for value in values],)
            self.x = torch.cat([self.x, values.unsqueeze(0)])
            self.y.append(label)
        content_io.close()

        cites_io = open(os.path.join(g.paths["data"], "cora", "cora.cites"))
        self.edge_index = torch.empty([0,], dtype=int)
        while True:
            line = cites_io.readline()
            if not line: break
            dst, src = line.split()
            edge = torch.tensor([int(node_ids.index(src)), int(node_ids.index(dst))], dtype=int)
            self.edge_index = torch.cat([self.edge_index, edge.unsqueeze(0)])
        self.edge_index = self.edge_index.t()
        cites_io.close()
        super(Cora, self).__init__()
