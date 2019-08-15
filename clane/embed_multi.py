import argparse

import torch
from torch import multiprocessing as mp

from graph import Graph
from models import MultiLayer, SingleLayer

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str)
parser.add_argument('--dim', type=int)
parser.add_argument('--cuda', type=int)
parser.add_argument('--multi', action='store_true')
parser.add_argument('--gamma', type=float)
config = parser.parse_args()
print(config)

device = torch.device(f'cuda:{config.cuda}')
G = Graph(
        dataset=config.dataset,
        directed=True,
        dim=config.dim,
        device=device
)

model_class = MultiLayer if config.multi else SingleLayer
prob_model = SingleLayer(dim=config.dim).to(device)

def train_transition_probability():
    pass

def g_standard():
    global G
    G.standardize()

@torch.no_grad()
def update_embedding(src_idx):
    global G
    nbrs = G.out_nbrs(src_idx)
    msg = prob_model.get_sims(G.standard_Z[src_idx], G.standard_Z[nbrs]).softmax(0)
    updated_embedding= G.X[src_idx] + torch.matmul(msg, G.Z[nbrs]).mul(config.gamma)
    return src_idx, updated_embedding 

    
if __name__ == '__main__':
    mp.set_start_method('spawn')
    
    while True:
        train_transition_probability()

        while True:
            previous_Z = G.Z.clone()
            with mp.Pool(processes=4, initializer=g_standard) as pool:
                for src_idx, embedding in pool.imap(update_embedding, range(len(G))):
                    G.Z[src_idx] = embedding
                    print(f'{src_idx}/{len(G)}', end='\r')
            print(torch.norm(G.Z - previous_Z,1))
            break
        
        # if embeddings are no more updated: break
        break
