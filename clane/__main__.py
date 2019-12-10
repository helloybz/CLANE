import argparse

import torch
from torch.utils.data import DataLoader

from clane.graphs.utils import get_graph
from clane.similarities.utils import get_similarity
from clane.loss import ApproximatedBCEWithLogitsLoss
from clane.utils import ContextManager


parser = argparse.ArgumentParser(prog='clane')
parser.add_argument('dataset', type=str)
parser.add_argument('similarity', type=str)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--gpu', type=str, default=None)
parser.add_argument('--gamma', type=float, default=0.76)
parser.add_argument('--tol_P', type=int, default=30)
parser.add_argument('--tol_Z', type=int, default=30)

config = parser.parse_args()

device = torch.device('cpu') if config.gpu is None \
    else torch.device(f'cuda:{config.gpu}')

manager = ContextManager(config)

# Prepare a graph.
g = get_graph(
    dataset=config.dataset,
)

while True:
    # TODO: Determine a condition of this loop ends.

    similarity = get_similarity(
        measure=config.similarity,
        dim=g.dim
    ).to(device)

    if not similarity.is_nonparametric:
        loss = ApproximatedBCEWithLogitsLoss(reduction='sum')
        optimizer = torch.optim.Adam(
            similarity.parameters(),
            lr=config.lr,
        )

        tolerence_P = config.tol_P
        epoch_P = 0

        while tolerence_P != 0:
            epoch_P += 1

            train_size = int(0.8*len(g))
            train_set, valid_set = torch.utils.data.random_split(
                g, [train_size, len(g)-train_size]
            )

            train_loader = DataLoader(
                dataset=train_set,
                batch_size=config.batch_size,
                drop_last=False,
                shuffle=True,
                pin_memory=True,
            )

            train_cost = 0
            for batch_index, (index_batch, edge_batch)\
                    in enumerate(train_loader):
                optimizer.zero_grad()

                z_batch = g.z[index_batch].to(device, non_blocking=True)
                edge_batch = edge_batch.to(device, non_blocking=True)

                z_srcs, z_dsts = z_batch.split(split_size=1, dim=1)
                cost = loss(similarity(z_srcs, z_dsts), edge_batch)
                cost.backward()
                optimizer.step()

                train_cost += float(cost)
                progress = 100*batch_index*config.batch_size/len(train_set)
                print(f'Epoch: {epoch_P} Train: {progress:.2f}% ',
                      end='\r')
            print(f'Epoch: {epoch_P:3d} '
                  + f'tol: {tolerence_P:3d} '
                  + f'Train loss: {train_cost:.2f} ')

            valid_loader = DataLoader(
                dataset=valid_set,
                batch_size=config.batch_size,
                drop_last=False,
                shuffle=True,
                pin_memory=True,
            )

            with torch.no_grad():
                valid_cost = 0
                for batch_index, (index_batch, edge_batch)\
                        in enumerate(valid_loader):
                    z_batch = g.z[index_batch].to(device, non_blocking=True)
                    edge_batch = edge_batch.to(device, non_blocking=True)

                    z_srcs, z_dsts = z_batch.split(split_size=1, dim=1)
                    cost = loss(similarity(z_srcs, z_dsts), edge_batch)

                    valid_cost += float(cost)
                    progress = 100*batch_index*config.batch_size/len(valid_set)
                    print(f'Epoch: {epoch_P} Valid: {progress:.2f}% ',
                          end='\r')
            print(f'Epoch: {epoch_P:3d} '
                  + f'tol: {tolerence_P:3d} '
                  + f'Valid loss: {valid_cost:.2f} ')

            manager.write('P/train loss', train_cost)
            manager.write('P/valid loss', valid_cost)

            init_required = manager.update_best_model(similarity, valid_cost)
            tolerence_P = config.tol_P if init_required else tolerence_P - 1

            # TODO: update lr
        manager.capture('P')

    tolerence_Z = config.tol_Z
    z_snapshot = g.z.clone()

    if not similarity.is_nonparametric:
        similarity = manager.get_best_model(model=similarity, device=device)

    epoch_Z = 0
    with torch.no_grad():
        while tolerence_Z != 0:
            epoch_Z += 1
            prev_z = g.z.clone()
            for idx in range(g.z.shape[0]):
                indices = g.edge_index[:, g.edge_index[0] == idx].t()
                idx_src, idx_dst = indices.split(split_size=1, dim=1)

                weight = similarity(z_snapshot[idx_src], z_snapshot[idx_dst])
                weight = weight.softmax(0)

                g.z[idx] = \
                    g.x[idx] + \
                    torch.matmul(
                        weight,
                        g.z[idx_dst].squeeze(1)
                    ).mul(config.gamma)

                progress = 100*(idx+1)/g.z.shape[0]
                print(f'Epoch: {epoch_Z}, Progress: {progress:.2f}% '
                      + f'tol: {tolerence_Z}', end='\r')

            distance = torch.norm(g.z - prev_z, 1)
            init_required = manager.update_best_model(g, float(distance))
            tolerence_Z = config.tol_Z if init_required else tolerence_Z - 1

            manager.write('Z', float(distance))
    manager.capture('Z')
    manager.steps['iter'] += 1
