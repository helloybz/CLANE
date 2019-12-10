import argparse
import os
import time

from numpy import inf
from tensorboardX import SummaryWriter
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.multiprocessing as mp

from graph import Graph
from models import MultiLayer, SingleLayer
from settings import PICKLE_PATH

    
def train_model(model, config, cost_list, cuda):
    start_time = time.time()
    G = Graph(
            dataset=config.dataset,
            device=torch.device(f'cuda:{config.cuda}')
    )
    
    optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.lr,
    )
    eps= 1e-6
    train_cost=0
    for z_src, z_nbrs, z_negs in G:
        if z_nbrs.shape[0] == 0: continue
        optimizer.zero_grad()
        nbr_probs = model(z_src, z_nbrs)
        nbr_probs = nbr_probs.masked_fill(nbr_probs==0, eps)
        nbr_nll = F.binary_cross_entropy(
                input=nbr_probs, 
                target=nbr_probs.new_ones(nbr_probs.shape),
                reduction='sum'
            )
         
        neg_probs = model(z_src, z_negs)
        neg_probs = neg_probs.masked_fill(neg_probs==1, 1-eps)
        if config.aprx:
            neg_probs = neg_probs.masked_select(neg_probs.bernoulli().byte())
        neg_nll = F.binary_cross_entropy(
                input=neg_probs,
                target=neg_probs.new_zeros(neg_probs.shape),
                reduction='sum',
            )
        loss= nbr_nll + neg_nll
        loss.backward()
        optimizer.step()
        train_cost += loss
    cost_list.append(train_cost.detach().item())


def update_embeddings(idx_q, cuda, cpu, Z, prev_Z, X, A, sim_func, gamma):
    with torch.no_grad():
        while True:
            idx = idx_q.get()
            if idx == None:
                idx_q.put(idx)
                break
            weight = sim_func(prev_Z[idx].to(cuda), prev_Z[A[idx]].to(cuda)).softmax(-1)
            embedding = X[idx].to(cuda) + torch.matmul(weight, prev_Z[A[idx]].to(cuda)).mul(gamma)
            Z[idx] = embedding.to(cpu)
            del idx, weight, embedding

def produce_q(idx_q, num_nodes):
    for idx in range(num_nodes):
        idx_q.put(idx)
    idx_q.put(None)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--directed', action='store_true')
    
    parser.add_argument('--aprx', action='store_true')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--decay', type=float, default=0.1)
    parser.add_argument('--stepsize', type=int, default=100)
    parser.add_argument('--tolerence_P', type=int, default=200)
    parser.add_argument('--valid_size', type=float, default=0.1)
    parser.add_argument('--n_workers_P', type=int, default=1)

    parser.add_argument('--gamma', type=float, default=0.74)
    parser.add_argument('--tolerence_Z', type=int, default=5)
    parser.add_argument('--n_workers_Z', type=int, default=1)
    
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--cuda', type=int)
    parser.add_argument('--tensorboard', action='store_true')
  
    config = parser.parse_args()
    print(config)
    # Build model tag.
    model_tag = f'{config.dataset}_clane_valid{config.valid_size}_g{config.gamma}_lr{config.lr}_decay{config.decay}'
    
    if config.aprx:
        model_tag += '_aprx'
    
    if config.directed:
        model_tag += '_directed'
    else:
        model_tag += '_undirected'
    
    if config.debug: breakpoint() 
    
    # Initialize the settings.
    mp.set_start_method('spawn')
    cuda = torch.device(f'cuda:{config.cuda}')
    cpu = torch.device(f'cpu')
    eps= 1e-6
    context = {
            'iteration': 0,
            'n_Z':0,
            'n_P':0
        }
    manager = mp.Manager()
    if config.tensorboard:
        writer = SummaryWriter(logdir='runs/{model_tag}')

    # Load Dataset(Graph) 
    G = Graph(
            dataset=config.dataset,
            directed=True,
        )

    G.Z.share_memory_()
    G.X.share_memory_()
    A = manager.dict(G.A)

    lr_copy = config.lr 
    while True:
        context['iteration'] += 1

        #
        # Train the transition probability model
        #
        min_cost = inf
        tolerence_P = config.tolerence_P
        
        # Instantiate a new model.
        prob_model = SingleLayer(dim=G.d).to(cuda)
        prob_model.share_memory()

        # reset learning rate.
        config.lr = lr_copy

        while True:
            context['n_P'] += 1
            start_time = time.time()
            cost_list = manager.list()
            processes = []
            
            if context['n_P'] % config.stepsize == 0:
                config.lr *= config.decay
            
            for rank in range(config.n_workers_P):
                p = mp.Process(
                        group=None,
                        target=train_model,
                        args=(prob_model, config, cost_list, cuda)
                    )
                processes.append(p)
            
            for p in processes:
                p.start()
            
            for p in processes:
                p.join()
             
            average_cost = sum(cost_list)/config.n_workers_P
            if min_cost > average_cost:
                min_cost = average_cost
                tolerence_P = config.tolerence_P
                torch.save(
                    prob_model,
                    os.path.join(PICKLE_PATH, f'{model_tag}_temp'))
            else:
                tolerence_P -= 1
            
            print(f'[Iter{context["iteration"]}-Epoch{context["n_P"]}]\ttrain_cost:{average_cost:10.5}\ttol:{tolerence_P:3}\tcollapsed:{time.time()-start_time:3.3}')
            if config.tensorboard:
                writer.add_scalars(
                    f'{model_tag}/model_training',
                    {'train_cost':average_cost},
                    context['n_P']                    
                )

            if tolerence_P == 0:
                prob_model = torch.load(
                    os.path.join(PICKLE_PATH, f'{model_tag}_temp'),
                    map_location=cuda)
                prob_model.share_memory()
                break
        

        #
        # Update Embeddings
        #
        tolerence_Z = config.tolerence_Z
        min_distance = inf
        
        if config.debug:
            idx_q.put(0)
            breakpoint()
            update_embeddings(idx_q, cuda, cpu, G.Z, G.Z.clone(), G.X, A, prob_model.get_sims, config.gamma)
        
        while True:
            start_time = time.time()
            context['n_Z'] += 1
            prev_Z = G.Z.clone().share_memory_()
            idx_q = mp.Queue()
            producers = []
            updators = []
            
            for rank in range(1):
                producers.append(
                        mp.Process(
                                target=produce_q, 
                                args=(idx_q, len(G))
                            )
                    )
            
            for rank in range(config.n_workers_Z):
                updator = mp.Process(
                        target=update_embeddings,
                        args=(idx_q, cuda, cpu,
                                G.Z, prev_Z, G.X, A, 
                                prob_model.get_sims, config.gamma),
                    )
                updators.append(updator)

            for producer in producers:
                producer.start()

            for updator in updators:
                updator.start()

            for producer in producers:
                producer.join()
            
            for updator in updators:
                updator.join()
            
            distance = torch.norm(G.Z-prev_Z,1)
            if min_distance > distance:
                min_distance = distance
                tolerence_Z = config.tolerence_Z
            else:
                tolerence_Z -= 1
            
            print(f'dist:{distance.item():5.3}\tmin_dist:{min_distance.item():5.3}\ttol:{tolerence_Z}\tcollaped:{time.time()-start_time}')
            
            if config.tensorboard:
                writer.add_scalars(
                    f'{model_tag}/embedding_updating',
                    {'distance':distance},
                    context['n_Z']                    
                )
            if tolerence_Z == 0:
                torch.save(
                        G.Z,
                        os.path.join(PICKLE_PATH, 'embeddings', f'{model_tag}_iter_{context["iteration"]}')
                    )
                break
