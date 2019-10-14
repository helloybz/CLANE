import argparse
import os

from numpy import inf
import torch

from graph import Graph
from manager import Manager
from misc import get_model_tag
from models import Similarity

from settings import PICKLE_PATH

parser = argparse.ArgumentParser()
parser.add_argument('--test_period', type=int, default=0, help='Period of the iterations to score the performance.')
parser.add_argument('--dataset', type=str, default='cora', help='Name of network dataset to embed. Default: \'cora\'.')
parser.add_argument('--reduce_factor', type=float, default=0.8, help='Amount of which the learning rate is reduced by.')
parser.add_argument('--lr', type=float, default=1e-5, help='Initial learning rate.')
parser.add_argument('--reduce_tol', type=int, default=30, help='Tolerance for lr reducing.')
parser.add_argument('--tol_P', type=int, default=100, help='Tolerance for training transition matrix P.')
parser.add_argument('--tol_Z', type=int, default=100, help='Tolerance for optimizing the embeddings Z.')
parser.add_argument('--aprx', action='store_true', default=False,
                    help='If True, approximate the loss while training P.')
parser.add_argument('--gamma', type=float, default=0.76, help='Gamma.')
parser.add_argument('--gpu', type=int, help='Index of GPU to use.')
config = parser.parse_args()


def train_transition_matrix(model, dataset, **settings):
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=settings['lr'],
    )
    breakpoint()
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer,
        factor=settings['reduce_factor'],
        patience=settings['reduce_tol'],
        threshold=0,
        verbose=True,
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        pin_memory=True,
    )
    device = settings['device']
    loss = torch.nn.BCELoss()
    tol = settings['tol']
    min_cost = inf
    import time
    start_time = time.time()

    while tol != 0:
        cost = 0
        for idx, (z_src, z_nbrs, z_negs) in enumerate(loader):
            if z_nbrs.shape[1] == 0: continue
            z_src = z_src.to(device, non_blocking=True)
            z_nbrs = z_nbrs.to(device, non_blocking=True)
            z_negs = z_negs.to(device, non_blocking=True)
            z_nbrs.squeeze_(0)
            z_negs.squeeze_(0)
            optimizer.zero_grad()

            sim_nbrs = model(z_src, z_nbrs)  # Compute the similarities between src node and its neighbors.
            probs_nbrs = sim_nbrs.sigmoid()  # Compute the (edge-existance) probabilities from 'sim_nbrs'.
            probs_nbrs = probs_nbrs.masked_fill(probs_nbrs == 0,
                                                1e-6)  # Replace 0 with epsilon (1e-6) to avoid exceptions.

            sim_negs = model(z_src, z_negs)
            probs_negs = sim_negs.sigmoid()
            probs_negs = probs_negs.masked_fill(probs_negs == 1, 1 - 1e-6)
            if settings['aprx']:
                # If config.aprx is True, do bernoulli trial to each probability and pick only ones whose trial comes 1.
                breakpoint()
                probs_negs = probs_negs.masked_select(probs_negs.bernoulli().byte())

            loss_nbrs = loss(probs_nbrs, probs_nbrs.new_ones(probs_nbrs.shape))
            loss_negs = loss(probs_negs, probs_negs.new_ones(probs_negs.shape))
            total_loss = loss_nbrs + loss_negs
            total_loss.backward()
            cost += total_loss.item()
            optimizer.step()
            print(idx, cost, end='\r')
        settings['manager'].log_result('P', cost)
        settings['manager'].increase_step_p()
        lr_scheduler.step(cost)
        if min_cost > cost:
            min_cost = cost
            tol = settings['tol']
        else:
            tol -= 1

        print(cost, tol, time.time() - start_time)


@torch.no_grad()
def update_embeddings(model, graph, **settings):
    tol = settings['tol']
    device = settings['device']
    min_distance = inf

    # Copy of Z to be used for computing transition probabilities.
    # Note that the transition probabilities should be fixed while optimizing the embeddings.
    initial_Z = graph.Z.clone()

    while tol != 0:
        # Another copy of Z to be used for computing how much the embeddings are updated during current iteration.
        prev_Z = graph.Z.clone()
        for node_idx in range(len(graph)):
            # Compute the similarities bewtween src node and its nbrs.
            # Because the transition matrix P should be fixed in optimizing phase, 'initial_Z' is used.
            init_z_src = initial_Z[node_idx].to(device)
            init_z_nbrs = initial_Z[graph.A[node_idx]].to(device)
            sim_nbrs = model(init_z_src, init_z_nbrs)

            # To ensure the convergence of the embeddings, the sum of the weights should be 1. (Transition matrix)
            weight_nbrs = sim_nbrs.softmax(0)

            z_nbrs = prev_Z[graph.A[node_idx]].to(device)

            # new_embedding =
            #     src node's internal content embeddings, X[src].
            #     + gamma * sum(weight * Current Z[nbrs])
            new_embedding = \
                graph.X[node_idx].to(device) \
                + torch.matmul(weight_nbrs, z_nbrs).mul(settings['gamma'])

            graph.Z[node_idx] = new_embedding.cpu()

        distance = torch.norm(graph.Z - prev_Z, 1)
        settings['manager'].log_result('z', distance)
        settings['manager'].increase_step_z()
        if distance == 0: return 0
        if min_distance > distance:
            min_distance = distance
            tol = settings['tol']
        else:
            tol -= 1
        print(distance, tol, end='\r')


if __name__ == '__main__':
    # Process manager
    manager = Manager(
        test_period=config.test_period,
        model_tag=get_model_tag(config),
    )

    # Load network dataset
    # TODO: Check where the dataset should be stored. CUDA or cpu.
    G = Graph(
        dataset=config.dataset
    )
    device = torch.device('cpu' if config.gpu is None else f'cuda:{config.gpu}')

    # Outer iteration.
    while True:
        # Similarity model should be initialized whenever the iteration starts.
        similarity = Similarity(G.dim).to(device)

        manager.increase_iter()

        # Train the transition matrix P with fixed node embedding Z.
        train_transition_matrix(
            model=similarity,
            dataset=G,
            **{'lr': config.lr,
               'reduce_factor': config.reduce_factor,
               'reduce_tol': config.reduce_tol,
               'tol': config.tol_P,
               'aprx': config.aprx,
               'device': device,
               'manager': manager,
               }
        )

        # Optimize the node embeddings Z, with the fixed transition matrix P.
        update_embeddings(
            model=similarity,
            graph=G,
            **{
                'gamma': config.gamma,
                'tol': config.tol_Z,
                'device': device,
                'manager': manager,
            }
        )

        print(manager.current_iteration)

        # save the embeddings
        breakpoint()
        torch.save(G.Z, os.path.join(PICKLE_PATH, 'embeddings', get_model_tag(config)))
        
        # Do simple node classification experiment to check the score.
        if manager.is_time_to_test():
            from experiments import NodeClassification

            tester = NodeClassification(
                X=G.Z,
                Y=G.Y,
                test_size=0.8
            )
            tester.train()
            print('Classification Test at {manager.current_iteration}')
            result = tester.test()
            for key in result:
                print(f'{key}: {result[key]}')
