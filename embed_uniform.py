import argparse
import os
import pdb
import pickle

from numpy import inf
from tensorboardX import SummaryWriter
from sklearn.metrics.pairwise import cosine_similarity
import torch
from torch.nn import functional as F

from dataset import CoraDataset
from settings import PICKLE_PATH


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str)
parser.add_argument('--load', type=str)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--gamma', type=float, default=0.74)
parser.add_argument('--tolerence_Z', type=int, default=30)
parser.add_argument('--model_tag', type=str)
config = parser.parse_args()

writer = SummaryWriter(log_dir='runs/{}'.format(config.model_tag))

def update_embedding(graph, sim_metric, recent_Z, gamma):
    prev_Z = graph.Z.clone()
    for src in range(graph.Z.shape[0]):
        nbrs = graph.out_nbrs(src)
        if nbrs.shape[0] == 0: continue
        sims = torch.ones(nbrs.shape).mul(1/nbrs.shape[0]).mul(gamma).cuda()
        graph.Z[src] = graph.X[src] + torch.matmul(sims.view(1,-1), prev_Z[nbrs].view(-1,graph.d))

    return torch.norm(graph.Z - prev_Z, 1)


if __name__ == "__main__":
    device = torch.device(f'cuda:{config.gpu}') if config.gpu is not None else torch.device('cpu')
    gamma = config.gamma
    tolerence_Z = config.tolerence_Z

    if config.dataset == 'cora':
        graph = CoraDataset(device=device, load=config.load or 'cora_X')
    else:
        raise ValueError

    import pdb; pdb.set_trace() 
    sim_metric = lambda x,y: F.cosine_similarity(x,y,dim=-1)

    min_distance = inf
    tolerence = tolerence_Z
    recent_Z = graph.Z.clone()

    context = {
        'iteration': 0,
        'n_Z': 0
    }

    while True:
        context['iteration'] += 1

        tolerence = config.tolerence_Z
        min_distance = inf

        recent_converged_Z = graph.Z.clone()
        while True:
            context['n_Z'] += 1
            distance = update_embedding(
                    graph,
                    sim_metric,
                    recent_converged_Z,
                    config.gamma
                )

            if min_distance > distance:
                min_distance = distance
                tolerence = config.tolerence_Z
            else:
                tolerence -= 1

            print(f'[EMBEDDING] {min_distance:5.5} tol:{tolerence}        ', end='\r')
            writer.add_scalars(
                    f'{config.model_tag}/embedding',
                    {'dist': distance},
                    context['n_Z']
                )
            if tolerence == 0:
                pickle.dump(
                        graph.Z.cpu().numpy(),
                        open(os.path.join(PICKLE_PATH, 'embedding',
                                f'{config.model_tag}_iter_{context["iteration"]}'
                            ), 'wb')
                    )
                break
