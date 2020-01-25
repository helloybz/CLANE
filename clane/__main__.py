import argparse

import torch
from torch.utils.data import DataLoader

from graphs.utils import get_graph
from similarities.utils import get_similarity
from loss import ApproximatedBCEWithLogitsLoss
from utils import ContextManager

# TODO: Maker better helps.
parser = argparse.ArgumentParser(prog='CLANE')
parser.add_argument('dataset', type=str,
    help='Name of the graph dataset.')
parser.add_argument('similarity', type=str)
parser.add_argument('iteration', type=int)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--gpu', type=str, default=None)
parser.add_argument('--gamma', type=float, default=0.76)
parser.add_argument('--tol_P', type=int, default=30)
parser.add_argument('--tol_Z', type=int, default=30)
parser.add_argument('--num_workers', type=int, default=0,
    help='The number of the workers for a dataloader.')

config = parser.parse_args()

device = torch.device('cpu') if config.gpu is None \
    else torch.device(f'cuda:{config.gpu}')

manager = ContextManager(config)

# Prepare a graph.
g = get_graph(
    dataset=config.dataset,
)

while manager.steps['iter'] != config.iteration:

    # TODO: Determine whether the simliarity measure
    #       need to be initialized per iterations.
    #       if NO, move the get_similarity method
    #       out of the while loop.      
    similarity = get_similarity(
        measure=config.similarity,
        dim=g.feature_dim
    ).to(device)
    
    g.node_traversal = False # to be False, while training the P.
    if not similarity.is_nonparametric:
        loss = ApproximatedBCEWithLogitsLoss(reduction='sum')

        optimizer = torch.optim.Adam(
            similarity.parameters(),
            lr=config.lr,
        )

        # TODO: these are supposed to be handled in ContextManger.
        tolerence_P = config.tol_P
        epoch_P = 0

        while tolerence_P != 0:
            epoch_P += 1

            dataloader = DataLoader(
                dataset=g,
                batch_size=config.batch_size,
                drop_last=False,
                shuffle=True,
                pin_memory=True,
                num_workers=config.num_workers
            )

            train_cost = 0
            for batch_index, (z_pairs, edges)\
                    in enumerate(dataloader):
                optimizer.zero_grad()
                z_pairs = z_pairs.to(device , non_blocking=True)
                edges = edges.to(device, non_blocking=True)

                sims = similarity(*z_pairs.split(split_size=1, dim=1))
                edge_probs = sims.sigmoid()
                cost = loss(edge_probs, edges)
                cost.backward()
                optimizer.step()

                train_cost += float(cost)
                progress = 100*batch_index*config.batch_size/len(g)
                print(f'Epoch: {epoch_P} Train: {progress:.2f}% ',
                      end='\r')
            print(f'Epoch: {epoch_P:3d} '
                  + f'tol: {tolerence_P:3d} '
                  + f'Train loss: {train_cost:.2f} ')

            manager.write('P/train loss', train_cost)

            init_required = manager.update_best_model(similarity, train_cost)
            tolerence_P = config.tol_P if init_required else tolerence_P - 1

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
