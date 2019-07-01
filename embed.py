import argparse
import os
import pdb
import pickle

from numpy import inf
from tensorboardX import SummaryWriter
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau, MultiStepLR
from torch.utils.data import DataLoader, Dataset

from dataset import CoraDataset, CiteseerDataset
from models import EdgeProbability
from settings import PICKLE_PATH
           

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='cora')
parser.add_argument('--load', type=str)
parser.add_argument('--sampled', action='store_true')

parser.add_argument('--tolerence_Z', type=int, default=30)
parser.add_argument('--approximated', action='store_true')
parser.add_argument('--gamma', type=float, default=0.74)

parser.add_argument('--tolerence_P', type=int, default=30)
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


def train_epoch(model, train_set, optimizer, approximated):
    model.train()
    eps=1e-6
    train_cost = 0
    for z_src, z_out, z_neg in train_set:
        optimizer.zero_grad()
        if z_out.shape[0] == 0: continue
        out_probs = model(z_src, z_out) 
        neg_probs = model(z_src, z_neg)
             
        out_loss = out_probs.where(
                out_probs!=0, 
                torch.ones(out_probs.shape, device=device).mul(eps)
            ).log().neg().sum()

        if approximated:
            neg_probs = neg_probs[neg_probs.bernoulli().byte()]
        neg_loss = (1-neg_probs).where(
                (1-neg_probs)!=0, 
                torch.ones(neg_probs.shape, device=device).mul(eps)
            ).log().neg().sum()
        total_loss = out_loss + neg_loss
        
        total_loss.backward()
        optimizer.step()
        train_cost += total_loss.item()
    return train_cost

@torch.no_grad()
def valid_epoch(model, valid_set, approximated):
    model.eval()
    eps=1e-10
    valid_cost = 0
    for z_src, z_out, z_neg in valid_set:
        if z_out.shape[0] == 0: continue
        out_probs = model(z_src, z_out) 
        neg_probs = model(z_src, z_neg)
        
        out_loss = out_probs.where(
                out_probs!=0,
                torch.ones(out_probs.shape, device=device).mul(eps),
            ).log().neg().sum()
        if approximated:
            neg_probs = neg_probs[neg_probs.bernoulli().byte()]
        neg_loss = (1-neg_probs).where(
                (1-neg_probs)!=0, 
                torch.ones(neg_probs.shape, device=device).mul(eps)
            ).log().neg().sum()
        
        total_loss = out_loss + neg_loss
        valid_cost += total_loss.item()
         
    return valid_cost

if __name__ == '__main__':
    config = parser.parse_args()
    print(config)

    device = torch.device('cuda:{}'.format(config.gpu))
    approximated = config.approximated
    writer = SummaryWriter(log_dir='runs/{}'.format(config.model_tag))
    eps=1e-6
   
    if config.dataset == 'cora':
        graph = CoraDataset(device=device, sampled=config.sampled, load=config.load or 'cora_X')
    elif config.dataset == 'citeseer':
        graph = CiteseerDataset(
                device=device,
                sampled=config.sampled,
                load=config.load or 'citeseer_X'
            )
    else:
        raise ValueError

    num_valid = int(config.valid_size * len(graph))
    num_train = len(graph)-num_valid
   
    context = {'iteration': 0,
            'n_P': 0,
            'n_Z': 0
        }

    while True:
        context['iteration'] += 1

        tolerence = config.tolerence_P
        min_valid_cost = inf

        model = EdgeProbability(dim=graph.Z.shape[1]).to(device)
        optimizer = torch.optim.Adam(
                model.parameters(), 
                lr=config.lr)
        graph.standard()
        lr_scheduler = MultiStepLR(
                optimizer,
                milestones=[100,180,250,500],
                gamma=0.1,
            )
        try:
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
                        graph, [num_train, num_valid])

                train_cost = train_epoch(model, train_set, optimizer, approximated)
                valid_cost = valid_epoch(model, valid_set, approximated)
                
                lr_scheduler.step()
                
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
                print(f'[MODEL TRAINING] {train_cost:5.5} {9*valid_cost:5.5} tol: {tolerence}                ', end='\r')
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
            graph.Z = torch.tensor(pickle.load(open(
                    os.path.join(
                            PICKLE_PATH, 'embedding',
                            f'{config.model_tag}_iter_{context["iteration"]}'),
                    'rb')
                )).to(device)
        except:
            graph.standard()
            recent_converged_Z = graph.standard_Z.clone()
            while True:
                context['n_Z'] += 1
                distance = update_embedding(
                        graph, 
                        model.get_sims, 
                        recent_converged_Z,
                        config.gamma, 
                    )
                
                if min_distance > distance:
                    min_distance = distance
                    tolerence = config.tolerence_Z
                else:
                    tolerence -= 1
                print(f'[EMBEDDING] {min_distance:5.5} tol:{tolerence}               ', end='\r')
                writer.add_scalars(f'{config.model_tag}/{"embedding"}',
                        {'dist': distance},
                        context['n_Z'] 
                    )
                if tolerence == 0: 
                    pickle.dump(
                            graph.Z.cpu().numpy(),
                            open(os.path.join(PICKLE_PATH, 'embedding', 
                                    f'{config.model_tag}_iter_{context["iteration"]}'
                                ), 'wb'))

                    print(f'[EMBEDDING] {min_distance:5.5}')
                    break

