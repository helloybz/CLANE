import argparse
import os
import time

from numpy import inf
from tensorboardX import SummaryWriter
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

from graph import Graph
from graph import collate
from models import MultiLayer, SingleLayer
from settings import PICKLE_PATH


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--directed', action='store_true')
    
    parser.add_argument('--aprx', action='store_true')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--decay', type=float, default=0.8)
    parser.add_argument('--tolerence_P', type=int, default=200)
    parser.add_argument('--valid_size', type=float, default=0.1)
    parser.add_argument('--batchsize', type=int, default=1)
     
    parser.add_argument('--gamma', type=float, default=0.74)
    parser.add_argument('--tolerence_Z', type=int, default=5)
     
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--cuda', type=int)
    
    config = parser.parse_args()
    print(config)
    # Build model tag.
    #TODO: Separate model_tag buidling function.
    model_tag = f'{config.dataset}_clane_valid{config.valid_size}_g{config.gamma}_lr{config.lr}_decay{config.decay}_batch{config.batchsize}'

    if config.aprx:
        model_tag += '_aprx'

    if config.directed:
        model_tag += '_directed'
    else:
        model_tag += '_undirected'
    
    if config.debug: breakpoint() 

    # Initialize the settings.
    cuda = torch.device(f'cuda:{config.cuda}')
    cpu = torch.device('cpu')
    eps= 1e-6
    G = Graph(
            dataset=config.dataset,
            directed=config.directed,
    )
    num_valid = int(len(G) * config.valid_size)
    num_train = len(G) - num_valid

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
        prob_model = SingleLayer(dim=G.d).to(cuda)
        optimizer = torch.optim.Adam(
                prob_model.parameters(),
                lr=config.lr,
        )
        lr_scheduler = ReduceLROnPlateau(
                optimizer,
                factor=config.decay,
                patience=30,
                verbose=True,
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
                    collate_fn=collate,
                    batch_size=config.batchsize,
                    drop_last=True,
                    num_workers=int(os.cpu_count()/2),
                    pin_memory=True,
                )
            train_cost = 0
             
            # Train
            prob_model.train()
            start_time= time.time()
            for idx, (z_src, z_nbrs, z_negs, nbr_mask, neg_mask) in enumerate(train_loader):
                batch_train_start = time.time()
                z_src = z_src.to(cuda, non_blocking=True)
                z_nbrs = z_nbrs.to(cuda, non_blocking=True)
                z_negs = z_negs.to(cuda, non_blocking=True)
                nbr_mask = nbr_mask.to(cuda, non_blocking=True)
                neg_mask = neg_mask.to(cuda, non_blocking=True)
                data_transfer_time = time.time() - batch_train_start
                
                optimizer.zero_grad()
                nbr_probs = prob_model(z_src, z_nbrs)
                nbr_probs = nbr_probs.masked_fill(nbr_probs.clone().detach()==0, eps)
                nbr_nll = nbr_probs.log().neg()
                nbr_cost = nbr_nll.mul(nbr_mask).sum()
                
                neg_probs = prob_model(z_src, z_negs)
                neg_probs = neg_probs.masked_fill(neg_probs.clone().detach()==1,1-eps)
                if config.aprx:
                    neg_mask = neg_mask.add(neg_probs.clone().detach().bernoulli())
                    neg_mask = neg_mask!=0
                neg_nll = (1-neg_probs).log().neg()
                neg_cost = neg_nll.mul(neg_mask.float()).sum()
                
                cost = nbr_cost + neg_cost
                cost = cost.div(config.batchsize)
                cost.backward()
                optimizer.step()
                train_cost += cost
                batch_train_end = time.time() - batch_train_start
            
            valid_cost = 0
            valid_loader = DataLoader(
                    valid_set, 
                    collate_fn=collate,
                    drop_last=False,
                    batch_size=config.batchsize,
                    num_workers=int(os.cpu_count()),
                    pin_memory=True
                )
            prob_model.eval()
            with torch.no_grad():
                for z_src, z_nbrs, z_negs, nbr_mask, neg_mask in valid_loader:
                    z_src = z_src.to(cuda, non_blocking=True)
                    z_nbrs = z_nbrs.to(cuda, non_blocking=True)
                    z_negs = z_negs.to(cuda, non_blocking=True)
                    nbr_mask = nbr_mask.to(cuda, non_blocking=True)
                    neg_mask = neg_mask.to(cuda, non_blocking=True)

                    probs = prob_model(z_src, z_nbrs)
                    nbr_probs = probs.masked_fill(probs==0, eps)
                    nbr_nll = nbr_probs.log().neg()
                    nbr_cost = nbr_nll.mul(nbr_mask).sum()

                    probs = prob_model(z_src, z_negs)
                    neg_probs = probs.masked_fill(probs==1, 1-eps)
                    neg_nll = (1-neg_probs).log().neg()
                    neg_cost = neg_nll.mul(neg_mask).sum()
                    # Should the approximation be skipped while validating?
#                    if config.aprx:
#                        neg_probs = neg_probs.masked_select(neg_probs.bernoulli().byte())

                    cost = nbr_cost + neg_cost
                    valid_cost += cost
          
#            valid_cost = valid_cost * (1-config.valid_size)/config.valid_size
            print(f'iter{context["iteration"]} - {context["n_P"]}\t' +
                    f'train cost:{train_cost:>10.5}\t' +
                    f'valid_cost:{valid_cost:>10.5}\t' + 
                    f'tol:{tolerence_P:3}\t' + 
                    f'elapsed:{time.time()-start_time:5.3}'
                )
            lr_scheduler.step(valid_cost)
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
                break
       
        # Embedding
        tolerence_Z = config.tolerence_Z
        min_distance = inf
        
        # Inner Iterations - Update the embeddings.
        while True:
            context['n_Z'] += 1
            previous_Z = G.Z.clone()
            loader = DataLoader(
                    G, 
                    num_workers=0 if config.debug else config.num_workers,
                    collate_fn=collate,
                )

            with torch.no_grad():
                for idx, (z_src, z_nbrs, __, nbr_mask, _) in enumerate(loader):
                    original_z_nbrs = G.Z[G.out_nbrs(idx)].to(cuda, non_blocking=True)
                    z_src = z_src.to(cuda, non_blocking=True)
                    z_nbrs = z_nbrs.to(cuda, non_blocking=True)
                    sims = prob_model.get_sims(z_src, z_nbrs).softmax(-1)
                    G.Z[idx] = G.X[idx] + torch.matmul(sims, original_z_nbrs).mul(config.gamma).cpu()

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
#        break
