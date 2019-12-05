import argparse

from numpy import inf
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
parser.add_argument('--gamma', type=float, default=0.76)
parser.add_argument('--tol_Z', type=int, default=30)

config = parser.parse_args()

device = torch.device('cpu') if config.gpu is None \
    else torch.device(f'cuda:{config.gpu}')

# Prepare a graph.
g = get_graph(
    dataset=config.dataset,
)

while True:
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

        node_pair_loader = DataLoader(
            dataset=g,
            batch_size=config.batch_size,
            drop_last=False,
            shuffle=True,
            pin_memory=True,
        )

        for batch_index, (index_batch, edge_batch) in enumerate(node_pair_loader):
            optimizer.zero_grad()
            z_batch = g.z[index_batch].to(device, non_blocking=True)
            edge_batch = edge_batch.to(device, non_blocking=True)
            srcs, dsts = z_batch.split(split_size=1, dim=1)
            cost = loss(similarity(srcs, dsts), edge_batch)
            cost.backward()
            optimizer.step()
            print(batch_index, cost)

    # Optimize the graph's node features.
    min_distance = inf
    tolerence_Z = config.tol_Z
    z_snapshot = g.z.clone()
    while tolerence_Z != 0:
        prev_z = g.z.clone()
        for idx in range(g.z.shape[0]):
            indice = g.edge_index[:, g.edge_index[0] == idx].t()
            idx_src, idx_dst = indice.split(split_size=1, dim=1)
            weight = similarity(z_snapshot[idx_src], z_snapshot[idx_dst])
            weight = weight.softmax(0)
            g.z[idx] = \
                g.x[idx] + \
                torch.matmul(
                    weight.squeeze(1),
                    g.z[idx_dst].squeeze(1)
                ).mul(config.gamma)

        # TODO: Measure how much the cla-embeddings are updated
        distance = torch.norm(g.z - prev_z, 1)
        if min_distance > distance:
            min_distance = distance
            tolerence_Z = config.tol_Z
        else:
            tolerence_Z -= 1
        print(distance, tolerence_Z)
