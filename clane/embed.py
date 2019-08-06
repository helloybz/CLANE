import argparse
import os
import pdb
import pickle

from numpy import inf
from tensorboardX import SummaryWriter
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau, MultiStepLR
from torch.utils.data import DataLoader, Dataset

from graph import Graph
import models
from settings import PICKLE_PATH
           

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='cora')
parser.add_argument('--directed', action='store_true')
parser.add_argument('--dim', type=int)

parser.add_argument('--tolerence_Z', type=int, default=30)
parser.add_argument('--gamma', type=float, default=0.74)

parser.add_argument('--tolerence_P', type=int, default=100)
parser.add_argument('--structure', type=str)
parser.add_argument('--aprx', action='store_true')
parser.add_argument('--valid_size', type=float, default=0.1)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--model_tag', type=str, default='test')

config = parser.parse_args()
print(config)

@torch.no_grad()
def update_embedding(graph, Pi, gamma):
    prev_Z = graph.Z.clone()
    for src in range(len(graph)):
        nbrs = graph.out_nbrs(src)
        graph.Z[src] = graph.X[src] + torch.matmul(Pi[src], graph.Z).mul(gamma)
    return torch.norm(graph.Z - prev_Z, 1)


def train_epoch(model, train_set, optimizer, aprx):
    model.train()
    eps=1e-6
    train_cost = 0
    for z_src, z_out, z_neg in train_set:
        optimizer.zero_grad()
        if z_out.shape[0] == 0: continue

        out_probs = model(z_src, z_out) 
        out_probs = out_probs.masked_fill(out_probs==0, eps)
        out_loss = out_probs.log().neg().sum()

        neg_probs = model(z_src, z_neg)
        if aprx:
            neg_probs = neg_probs.masked_select(neg_probs.bernoulli().byte())
        neg_probs = neg_probs.masked_fill(neg_probs==1, 1-eps)
        neg_loss = (1-neg_probs).log().neg().sum()
        total_loss = out_loss + neg_loss
        
        total_loss.backward()
        optimizer.step()
        train_cost += total_loss.item()
    return train_cost

@torch.no_grad()
def valid_epoch(model, valid_set, aprx):
    model.eval()
    eps=1e-6
    valid_cost = 0
    for z_src, z_out, z_neg in valid_set:
        if z_out.shape[0] == 0: continue
        out_probs = model(z_src, z_out) 
        out_probs = out_probs.masked_fill(out_probs==0, eps)
        out_loss = out_probs.log().neg().sum()

        neg_probs = model(z_src, z_neg)
        if aprx:
            neg_probs = neg_probs.masked_select(neg_probs.bernoulli().byte())
        neg_probs = neg_probs.masked_fill(neg_probs==1, 1-eps)
        neg_loss = (1-neg_probs).log().neg().sum()
        total_loss = out_loss + neg_loss
        valid_cost += total_loss.item()
         
    return valid_cost


device = torch.device('cuda:{}'.format(config.gpu))
writer = SummaryWriter(log_dir='runs/{}'.format(config.model_tag))
eps=1e-6

g = Graph(
        dataset=config.dataset,
        directed=config.directed,
        dim=config.dim,
        device=device
)
num_valid = int(config.valid_size * len(g))
num_train = len(g)-num_valid
context = {
    'iteration': 0,
    'n_P': 0,
    'n_Z': 0
}
model_class = models.MultiLayer if config.structure == 'multilayer' else models.SingleLayer

while True:
    context['iteration'] += 1
    
    # Settings for model training
    min_valid_cost = inf
    
    model = model_class(dim=g.d).to(device)
    optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=config.lr)
    g.standardize()
    lr_scheduler = ReduceLROnPlateau(
            optimizer,
            factor=0.9,
            patience=15,
            verbose=True,
            eps=0,
        )
    while True:
        context['n_P'] += 1

        train_set, valid_set = torch.utils.data.random_split(
                g, [num_train, num_valid])
        train_cost = train_epoch(model, train_set, optimizer, config.aprx)
        valid_cost = valid_epoch(model, valid_set, config.aprx)
        
        lr_scheduler.step(valid_cost)
        
        if min_valid_cost > valid_cost:
            min_valid_cost = valid_cost
            tolerence = config.tolerence_P
            torch.save(model,
                    os.path.join(PICKLE_PATH, 'models', f'{config.model_tag}_iter_{context["iteration"]}_temp'))
        else:
            tolerence -= 1

        writer.add_scalars(f'{config.model_tag}/{"model_training"}',
                {'train_cost': train_cost, 
                 'valid_cost': valid_cost*9},
                context['n_P'] 
            )
        print(f'[MODEL TRAINING] train error: {train_cost:5.5} validation error: {9*valid_cost:5.5} epoch: {context["n_P"]}, tolerence: {tolerence:3}', end='\r')
        if tolerence == 0: 
            model = torch.load(
                    os.path.join(PICKLE_PATH, 'models', 
                            f'{config.model_tag}_iter_{context["iteration"]}_temp'
                    ),
                    map_location=device
                )
            torch.save(model,
                    os.path.join(PICKLE_PATH, 'models', f'{config.model_tag}_iter_{context["iteration"]}'))
            print(f'[MODEL TRAINING] {train_cost:5.5} {9*valid_cost:5.5} tol: {tolerence}')
            break


    ## Embedding ##

    tolerence = config.tolerence_Z
    min_distance = inf

    # compute transition matrix.
    Pi = g.A.clone()
    for src in range(len(g)):
        probs = model(g.standard_Z[src], g.standard_Z[g.out_nbrs(src)]).softmax(0)
        for idx, prob in zip(g.out_nbrs(src),probs):
            Pi[src,idx] = prob
    
    # Generate the embeddings.
    while True:
        context['n_Z'] += 1
        distance = update_embedding(
                g, 
                Pi,
                config.gamma, 
            )
        
        if min_distance > distance:
            min_distance = distance
            tolerence = config.tolerence_Z
        else:
            tolerence -= 1
        
        print(f'[EMBEDDING] distance:{distance:8.5} tol:{tolerence:2}', end='\r')
        writer.add_scalars(f'{config.model_tag}/{"embedding"}',
                {'dist': distance},
                context['n_Z'] 
            )
        if tolerence == 0:
            torch.save(
                    g.Z.cpu(),
                    os.path.join(
                            PICKLE_PATH,
                            'embeddings', 
                            f'{config.model_tag}_iter_{context["iteration"]}'
                        )
                )
            break

