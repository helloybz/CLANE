import argparse

import torch
from torch.utils.data import DataLoader

from clane.graphs.utils import get_graph


parser = argparse.ArgumentParser(prog='clane')
parser.add_argument('dataset', type=str)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--gpu', type=str, default=None)
config = parser.parse_args()

device = torch.device('cpu') if config.gpu is None \
    else torch.device(f'cuda:{config.gpu}')

# Prepare a graph.
g = get_graph(
    dataset=config.dataset,
)
g.z = g['x'].clone()

edge_loader = DataLoader(
    dataset=g['edge_index'].t(),
    batch_size=config.batch_size,
    drop_last=False,
    shuffle=True,
    pin_memory=True,
)
# Prepare similarity measure.

# Train the parameters in the similarity measure.
for index_batch in edge_loader:
    print(g.z[index_batch].to(device, non_blocking=True))

# Optimize the graph's node features.
