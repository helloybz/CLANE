import argparse

import torch
from torch.utils.data import DataLoader

from clane.graphs.utils import get_graph
from clane.similarities.utils import get_similarity
from clane.loss import ApproximatedBCEWithLogitsLoss


parser = argparse.ArgumentParser(prog='clane')
parser.add_argument('dataset', type=str)
parser.add_argument('similarity', type=str)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--gpu', type=str, default=None)
config = parser.parse_args()

device = torch.device('cpu') if config.gpu is None \
    else torch.device(f'cuda:{config.gpu}')

# Prepare a graph.
g = get_graph(
    dataset=config.dataset,
)

node_pair_loader = DataLoader(
    dataset=g,
    batch_size=config.batch_size,
    drop_last=False,
    shuffle=True,
    pin_memory=True,
)

# Prepare similarity measure.
similarity = get_similarity(
    measure=config.similarity,
    dim=g.dim
).to(device)

# Train the parameters in the similarity measure.
if not similarity.is_nonparametric:
    loss = ApproximatedBCEWithLogitsLoss(reduction='sum')
    optimizer = torch.optim.Adam(
        similarity.parameters(),
        lr=config.lr,
    )

    for batch_index, (index_batch, edge_batch) in enumerate(node_pair_loader):
        optimizer.zero_grad()
        z_batch = g.z[index_batch].to(device, non_blocking=True)
        srcs, dsts = z_batch.split(split_size=1, dim=1)
        cost = loss(similarity(srcs, dsts), edge_batch)
        cost.backward()
        optimizer.step()
        print(batch_index, cost)

# Optimize the graph's node features.
