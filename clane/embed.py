import argparse

from numpy import inf
import torch

from graph import Graph
from manager import Manager
from models import Similarity


parser = argparse.ArgumentParser()
parser.add_argument('--test_period', type=int, default=0)
parser.add_argument('--dataset', type=str)
parser.add_argument('--reduce_factor', type=float, default=0.8)
parser.add_argument('--lr', type=float, default=1e-5)
parser.add_argument('--reduce_tol', type=int, default=30)
parser.add_argument('--tol_P', type=int, default=100)
parser.add_argument('--tol_Z', type=int, default=100)
parser.add_argument('--aprx', action='store_true', default=False)
parser.add_argument('--gamma', type=float, default=0.76)
parser.add_argument('--gpu', type=int)
config = parser.parse_args()

def train_transition_matrix(model, dataset, **settings):
    optimizer = torch.optim.Adam(
            model.parameters(),
            lr=settings['lr'],
    )
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            factor=settings['reduce_factor'],
            patience=settings['reduce_tol'],
            verbose=True,
    )
    loader = torch.utils.data.DataLoader(
        graph, 
        pin_memory=True,
    )
    device = settings['device']
    loss = torch.nn.BCELoss()
    tol = settings['tol']
    min_cost = inf
    epoch_counter = 0
    import time; start_time = time.time()

    while tol != 0:
        epoch_counter += 1
        cost = 0
        for idx, (z_src, z_nbrs, z_negs) in enumerate(loader):
            if z_nbrs.shape[1] == 0: continue
            z_src = z_src.to(device, non_blocking=True)
            z_nbrs = z_nbrs.to(device, non_blocking=True)
            z_negs = z_negs.to(device, non_blocking=True)
            z_nbrs.squeeze_(0)
            z_negs.squeeze_(0)
            optimizer.zero_grad()
            sim_nbrs = model(z_src, z_nbrs)
            probs_nbrs = sim_nbrs.sigmoid()
            probs_nbrs = probs_nbrs.masked_fill(probs_nbrs==0, 1e-6)

            sim_negs = model(z_src, z_negs)
            probs_negs = sim_negs.sigmoid()
            probs_negs = probs_negs.masked_fill(probs_negs==1, 1-1e-6)
            if settings['aprx']:
                probs_negs = probs_negs.masked_select(probs_negs.bernoulli().byte())

            loss_nbrs = loss(probs_nbrs, probs_nbrs.new_ones(probs_nbrs.shape))
            loss_negs = loss(probs_negs, probs_negs.new_ones(probs_negs.shape))
            total_loss =loss_nbrs + loss_negs
            total_loss.backward()
            cost += total_loss.item()
            optimizer.step()
            print(idx, cost, end='\r')
        lr_scheduler.step(cost)
        if min_cost > cost:
            min_cost = cost
            tol = settings['tol']
        else:
            tol -= 1
        
        print(cost, tol, time.time() - start_time)
    return epoch_counter

@torch.no_grad()
def update_embeddings(model, graph, **settings):
    tol = settings['tol']
    device = settings['device']
    min_distance = inf
    initial_Z = graph.Z.clone() # used for compute transition probs.
    while tol != 0:
        prev_Z= graph.Z.clone()
        for node_idx in range(len(graph)):
            z_src = prev_Z[node_idx].to(device)
            z_nbrs = prev_Z[graph.A[node_idx]].to(device)
            init_z_src = initial_Z[node_idx].to(device)
            init_z_nbrs = initial_Z[graph.A[node_idx]].to(device)
            weights = model(init_z_src, init_z_nbrs).softmax(0)
            new_embedding = \
                graph.X[node_idx].to(device) \
                + torch.matmul(weights, z_nbrs).mul(settings['gamma'])
            graph.Z[node_idx] = new_embedding.cpu()
        distance = torch.norm(graph.Z - prev_Z, 1)
        if distance == 0: return 0
        if min_distance > distance:
            min_distance = distance
            tol = settings['tol']
        else:
            tol -= 1
        print(distance, tol, end='\r')
    return 0


if __name__ == '__main__':
    # Outer Iteration
    manager = Manager(
        test_period = config.test_period 
    )
    graph = Graph(
        dataset = config.dataset
    )
    device = torch.device('cpu' if config.gpu is None else f'cuda:{config.gpu}')
    
    while True: 
        similarity = Similarity(graph.dim).to(device) 
        manager.increase_iter()
        
        train_transition_matrix(
            model = similarity,
            dataset = graph,
            **{'lr': config.lr,
             'reduce_factor': config.reduce_factor,
             'reduce_tol': config.reduce_tol,
             'tol': config.tol_P,
             'aprx': config.aprx,
             'device': device
            }
        )
        update_embeddings(
            model = similarity,
            graph = graph,
            **{
                'gamma': config.gamma,
                'tol': config.tol_Z,
                'device': device
            }
        )
        
        print(manager.current_iteration)
        if manager.is_time_to_test():
            from experiments import NodeClassification
            tester = NodeClassification(
                X = graph.Z,
                Y = graph.Y,
                test_size = 0.8
            )
            tester.train()
            print('Classification Test at {manager.current_iteration}')
            result = tester.test()
            for key in result:
                print(f'{key}: {result[key]}')

