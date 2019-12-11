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
parser.add_argument('iteration', type=int)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--lr_factor', type=float)
parser.add_argument('--lr_patience', type=int)
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

while manager.steps['iter'] != config.iteration:

    similarity = get_similarity(
        measure=config.similarity,
        dim=g.feature_dim
    ).to(device)
    g.node_traversal = False
    if not similarity.is_nonparametric:
        loss = ApproximatedBCEWithLogitsLoss(reduction='sum')

        optimizer = torch.optim.Adam(
            similarity.parameters(),
            lr=config.lr,
        )

        if config.lr_factor:
            lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer=optimizer,
                factor=config.lr_factor,
                patience=config.lr_patience,
                verbose=True,
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
            for batch_index, (z_pairs, edges)\
                    in enumerate(train_loader):
                optimizer.zero_grad()
                z_pairs = z_pairs.to(device, non_blocking=True)
                edges = edges.to(device, non_blocking=True)

                sims = similarity(*z_pairs.split(split_size=1, dim=1))
                edge_probs = sims.sigmoid()
                cost = loss(edge_probs, edges)
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
                for batch_index, (z_pairs, edges)\
                        in enumerate(valid_loader):
                    z_pairs = z_pairs.to(device, non_blocking=True)
                    edges = edges.to(device, non_blocking=True)

                    sims = similarity(*z_pairs.split(split_size=1, dim=1))
                    edge_probs = sims.sigmoid()
                    cost = loss(edge_probs, edges)

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
            if config.lr_factor:
                lr_scheduler.step(valid_cost)
        manager.capture('P')

    tolerence_Z = config.tol_Z
    g.node_traversal = True
    g.build_transition_matrix(similarity)

    if not similarity.is_nonparametric:
        similarity = manager.get_best_model(model=similarity, device=device)

    epoch_Z = 0
    with torch.no_grad():
        while tolerence_Z != 0:
            epoch_Z += 1
            distance = 0

            for idx, (x, z, w_nbrs, z_nbrs) in enumerate(g):
                x = x.to(device, non_blocking=True)
                z = z.to(device, non_blocking=True)
                w_nbrs = w_nbrs.to(device, non_blocking=True).softmax(0)
                z_nbrs = z_nbrs.to(device, non_blocking=True)
                new_z = x + torch.matmul(w_nbrs, z_nbrs).mul(config.gamma)
                distance += torch.norm(new_z - z, 1)
                g.set_z(idx, new_z)
                progress = 100*(idx+1)/g.z.shape[0]
                print(f'Epoch: {epoch_Z}, Progress: {progress:.2f}% '
                      + f'tol: {tolerence_Z}', end='\r')

            init_required = manager.update_best_model(g, float(distance))
            tolerence_Z = config.tol_Z if init_required else tolerence_Z - 1

            manager.write('Z', float(distance))
    manager.capture('Z')
    manager.steps['iter'] += 1
