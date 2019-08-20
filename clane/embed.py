import argparse
import os
import time

from numpy import inf
from tensorboardX import SummaryWriter
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch import multiprocessing as mp

from graph import Graph
from models import MultiLayer, SingleLayer
from settings import PICKLE_PATH

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--dim', type=int)
    parser.add_argument('--directed', action='store_true')
    
    parser.add_argument('--multi', action='store_true')
    parser.add_argument('--aprx', action='store_true')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--tolerence_P', type=int, default=60)
    parser.add_argument('--valid_size', type=float, default=0.1)
   
    parser.add_argument('--gamma', type=float)
    parser.add_argument('--tolerence_Z', type=int, default=5)
    
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--cuda', type=int)
    parser.add_argument('--num_workers', type=int, default=0)
    
    config = parser.parse_args()
    print(config)

    # Build model tag.
    model_tag = f'{config.dataset}_clane_d{config.dim}_g{config.gamma}_lr{config.lr}'

    if config.multi:
        model_tag += '_multi'
    else: 
        model_tag += '_single'

    if config.aprx:
        model_tag += '_aprx'

    if config.directed:
        model_tag += '_directed'
    else:
        model_tag += '_undirected'
    
    # Initialize the settings.
    cuda = torch.device(f'cuda:{config.cuda}')
    cpu = torch.device('cpu')
    eps= 1e-6
    mp.set_start_method('spawn')
    G = Graph(
            dataset=config.dataset,
            directed=config.directed,
            dim=config.dim,
            device=cuda
    )
    num_valid = int(len(G) * config.valid_size)
    num_train = len(G) - num_valid

    model_class = MultiLayer if config.multi else SingleLayer
    prob_model = model_class(dim=config.dim).to(cuda)
    prob_model.share_memory()

    context = {
        'iteration':0,
        'n_Z':0,
        'n_P':0
        }

    # Load TensorboardX
    writer = SummaryWriter(log_dir=f'runs/{model_tag}')


@torch.no_grad()
def update_embedding(G, src_idx, gamma):
    nbrs = G.out_nbrs(src_idx)
    msg = prob_model.get_sims(G.standard_Z[src_idx], G.standard_Z[nbrs]).softmax(0)
    updated_embedding= G.X[src_idx] + torch.matmul(msg, G.Z[nbrs]).mul(config.gamma)
    return src_idx, updated_embedding 


if __name__ == '__main__':
    # Outer iteration
    while True:
        context['iteration'] += 1

        tolerence_P = config.tolerence_P
        G.standardize()
        optimizer = torch.optim.Adam(
                prob_model.parameters(),
                lr=config.lr
        )
        lr_scheduler = ReduceLROnPlateau(
                optimizer,
                factor=0.1,
                patience=10,
                verbose=True,
                eps=0
        )
        min_cost = inf
        # Inner iteration 1 - Train transition matrix
        while True:
            context['n_P'] += 1

            # Preparation
            train_set, valid_set = torch.utils.data.random_split(
                    G, 
                    [num_train, num_valid]
                )
            
            train_loader = DataLoader(
                    train_set, 
                    num_workers=0 if config.debug else config.num_workers
                )
            train_cost = 0
            
            # Train
            prob_model.train()
            for z_src, z_out, z_neg in train_loader:
                optimizer.zero_grad()
                z_out = z_out.squeeze(0)
                z_neg = z_neg.squeeze(0)
                probs = prob_model(z_src, z_out)
                out_probs = probs.masked_fill(probs==0, eps)
                out_cost = out_probs.log().neg().sum()

                probs = prob_model(z_src, z_neg)
                neg_probs = probs.masked_fill(probs==1, 1-eps)
                if config.aprx:
                    neg_probs = neg_probs.masked_select(neg_probs.bernoulli().byte())
                neg_cost = (1-neg_probs).log().neg().sum()

                cost = out_cost + neg_cost
                cost.backward()
                optimizer.step()
                train_cost += cost
            
            valid_cost = 0
            valid_loader = DataLoader(
                    valid_set, 
                    num_workers=0 if config.debug else config.num_workers
                )

            prob_model.eval()
            with torch.no_grad():
                for z_src, z_out, z_neg in valid_loader:
                    z_out = z_out.squeeze(0)
                    z_neg = z_neg.squeeze(0)
                    probs = prob_model(z_src, z_out)
                    out_probs = probs.masked_fill(probs==0, eps)
                    out_cost = out_probs.log().neg().sum()

                    probs = prob_model(z_src, z_neg)
                    neg_probs = probs.masked_fill(probs==1, 1-eps)

                    # Should the approximation be skipped while validating?
                    if config.aprx:
                        neg_probs = neg_probs.masked_select(neg_probs.bernoulli().byte())

                    neg_cost = (1-neg_probs).log().neg().sum()

                    cost = out_cost + neg_cost
                    valid_cost += cost
           
            print(f'iter{context["iteration"]} - {context["n_P"]}\t' +
                    f'train cost:{train_cost:10.3}\t' +
                    f'valid_cost:{valid_cost:10.3}\t' + 
                    f'tol:{tolerence_P:3}'
                )
            lr_scheduler.step(valid_cost * (1-config.valid_size)*10)

            # Log on Tensorboard  
            writer.add_scalars(
                    f'{model_tag}/{"model_training"}',
                    {'train_cost': train_cost,
                     'valid_cost': valid_cost
                     },
                    context['n_P']
            )

            if min_cost > valid_cost:
                min_cost = valid_cost
                tolerence_P = config.tolerence_P
                # Save the best model
                torch.save(
                        prob_model,
                        os.path.join(PICKLE_PATH, f'{model_tag}_temp')
                    )
            else:
                tolerence_P -= 1

            if tolerence_P == 0:
                # Load the best model and break the iteration
                prob_model = torch.load(
                        os.path.join(PICKLE_PATH, f'{model_tag}_temp'),
                        map_location = cuda
                )
                prob_model.share_memory()
                break
       
        # Embedding
        tolerence_Z = config.tolerence_Z
        min_distance = inf

        # Inner Iterations - Update the embeddings.
        while True:
            context['n_Z'] += 1
            previous_Z = G.Z.clone()
            loader = DataLoader(
                    range(len(G)), 
                    num_workers=0 if config.debug else config.num_workers
                )

            with torch.no_grad():
                for idx in loader:
                    nbrs = G.out_nbrs(int(idx))
                    if nbrs.shape[0] == 0: continue
                    msg = prob_model.get_sims(G.standard_Z[idx].squeeze(0), G.standard_Z[nbrs]).softmax(0)
                    G.Z[idx] = G.X[idx] + torch.matmul(msg, G.Z[nbrs]).mul(config.gamma)
                distance = torch.norm(G.Z - previous_Z, 1)
                writer.add_scalars(
                        f'{model_tag}/{"embedding"}',
                        {"dist":distance},
                        context['n_Z']
                )
                print(distance, tolerence_Z) 
            if min_distance <= distance:
                tolerence_Z -= 1
            else:
                min_distance = distance

            if tolerence_Z == 0:
                torch.save(
                        G.Z.cpu(),
                        os.path.join(PICKLE_PATH, 'embeddings', f'{model_tag}_iter_{context["iteration"]}')
                    )
                break

        # if embeddings are no more updated: break
        break
