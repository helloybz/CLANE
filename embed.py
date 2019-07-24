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
from models import EdgeProbability
from settings import PICKLE_PATH
           

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='cora')
parser.add_argument('--directed', action='store_true')
parser.add_argument('--load', type=str)
parser.add_argument('--dim', type=int)

parser.add_argument('--tolerence_Z', type=int, default=30)
parser.add_argument('--gamma', type=float, default=0.74)

parser.add_argument('--tolerence_P', type=int, default=50)
parser.add_argument('--aprx', action='store_true')
parser.add_argument('--valid_size', type=float, default=0.1)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--model_tag', type=str, default='test')


@torch.no_grad()
def update_embedding(graph, sim_metric, recent_Z, gamma):
    prev_Z = graph.Z.clone()
    for src in range(graph.Z.shape[0]):
        nbrs = graph.out_nbrs(src)
        if nbrs.shape[0] == 0 : continue
        sims = sim_metric(recent_Z[src], recent_Z[nbrs])
        sims = sims.relu().add(eps)
        if sims.sum()==0: pdb.set_trace()

        sims = sims.div(sims.sum()).mul(gamma)
#        sims = sims.div(recent_Z[nbrs].norm(dim=1).mul(recent_Z[src].norm())).softmax(0)
        graph.Z[src] = graph.X[src] + torch.matmul(sims.view(1,-1), prev_Z[nbrs].view(-1,graph.d))
#        if float('nan') in graph.Z[src]:
#            print('efef')
#            pdb.set_trace()

    return torch.norm(graph.Z - prev_Z, 1)


def train_epoch(model, train_set, optimizer, aprx):
    model.train()
    eps=1e-6
    train_cost = 0
    for z_src, z_out, z_neg in train_set:
        optimizer.zero_grad()
        if z_out.shape[0] == 0: continue

        out_probs = model(z_src, z_out) 
        neg_probs = model(z_src, z_neg)
        
        out_probs = out_probs.masked_fill(out_probs==0, eps)
        out_loss = out_probs.log().neg().sum()

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
        neg_probs = model(z_src, z_neg)
        
        out_probs = out_probs.masked_fill(out_probs==0, eps)
        out_loss = out_probs.log().neg().sum()

        if aprx:
            neg_probs = neg_probs.masked_select(neg_probs.bernoulli().byte())
        neg_probs = neg_probs.masked_fill(neg_probs==1, 1-eps)
        neg_loss = (1-neg_probs).log().neg().sum()
        total_loss = out_loss + neg_loss
        valid_cost += total_loss.item()
         
    return valid_cost

if __name__ == '__main__':
    config = parser.parse_args()
    print(config)

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

    while True:
        context['iteration'] += 1

        # Settings for model training
        min_valid_cost = inf
        model = EdgeProbability(dim=g.d).to(device)
        optimizer = torch.optim.Adam(
                model.parameters(), 
                lr=config.lr)
        g.standardize()
#        lr_scheduler = MultiStepLR(
#                optimizer,
#                milestones=[100,180,250,500],
#                gamma=0.1,
#            )
        lr_scheduler = ReduceLROnPlateau(
                optimizer,
                verbose=True,
                eps=0,
            )
        try:
            # if a pickle exists, skip training.
            model = torch.load(
                    os.path.join(
                            PICKLE_PATH, 'models',
                            f'{config.model_tag}_iter_{context["iteration"]}'),
                    map_location=device
                )
        except:
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

        tolerence = config.tolerence_Z
        min_distance = inf

        try:
            g.Z = torch.tensor(pickle.load(open(
                    os.path.join(
                            PICKLE_PATH, 'embedding',
                            f'{config.model_tag}_iter_{context["iteration"]}'),
                    'rb')
                )).to(device)
        except:
            g.standardize()
            recent_converged_Z = g.standard_Z.clone()
            while True:
                context['n_Z'] += 1
                distance = update_embedding(
                        g, 
                        model.get_sims, 
                        recent_converged_Z,
                        config.gamma, 
                    )
                
                if min_distance > distance:
                    min_distance = distance
                    tolerence = config.tolerence_Z
                else:
                    tolerence -= 1
                print(f'[EMBEDDING] tol:{tolerence}', end='\r')
                writer.add_scalars(f'{config.model_tag}/{"embedding"}',
                        {'dist': distance},
                        context['n_Z'] 
                    )
                if tolerence == 0: 
                    torch.save(g.Z, f'{config.model_tag}_iter_{context["iteration"]}')
                    pickle.dump(
                            g.Z.cpu().numpy(),
                            open(os.path.join(PICKLE_PATH, 'embedding', 
                                    f'{config.model_tag}_iter_{context["iteration"]}'
                                ), 'wb'))

                    print(f'[EMBEDDING] {min_distance:5.5}')
                    break

