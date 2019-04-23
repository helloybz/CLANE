import argparse
import logging
import os
import pdb
import pickle

from numpy import inf
from tensorboardX import SummaryWriter
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset

from dataset import CoraDataset, CiteseerDataset
from experiment import node_classification
from models import EdgeProbability
from settings import PICKLE_PATH
           

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='cora')
parser.add_argument('--deepwalk', action='store_true')
parser.add_argument('--sampled', action='store_true')
parser.add_argument('--gamma', type=float, default=0.74)
parser.add_argument('--tolerence_Z', type=int, default=30)
parser.add_argument('--sim_metric', type=str)
parser.add_argument('--tolerence_P', type=int, default=30)
parser.add_argument('--test_size', type=float, default=0.1)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--model_tag', type=str, default='test')
config = parser.parse_args()
print(config)


logging.basicConfig(format='[%(levelname)s] %(message)s',level=logging.DEBUG)
writer = SummaryWriter(log_dir='runs/{}'.format(config.model_tag))

device = torch.device('cuda:{}'.format(config.gpu))

# -----------
#   Graph
# -----------
graph = CoraDataset(device=device, sampled=config.sampled, deepwalk=config.deepwalk)

# -----------
#   Model
# -----------
if config.sim_metric == 'edgeprob':
    prob_model = EdgeProbability(dim=graph.Z.shape[1]).to(device)

    optimizer = torch.optim.Adam(
            prob_model.parameters(), 
            lr=config.lr)
    lr_scheduler = ReduceLROnPlateau(optimizer, 'min', verbose=True)

context = {'iteration': 0,
       'n_P': 0,
       'n_Z': 0
   }


# ---------------
#   Sim metric
# ---------------
if config.sim_metric == 'edgeprob':
    sim_metric = prob_model.get_sims
elif config.sim_metric == 'cosine':
    sim_metric = lambda z,u: torch.nn.CosineSimilarity(dim=-1)(z,u).add(1).div(2)
else:
    raise ValueError


@torch.no_grad()
def update_embedding(graph, sim_metric, gamma, ):
    prev_Z = graph.Z.clone()
    for src in range(graph.Z.shape[0]):
        nbrs = graph.nbrs(src)
        if nbrs.shape[0] == 0: continue
        sims = sim_metric(prev_Z[src], prev_Z[nbrs])
        sims = sims.softmax(0)
        graph.Z[src] = graph.X[src] + torch.matmul(sims, prev_Z[nbrs]).mul(gamma)
    
    return torch.norm(graph.Z - prev_Z, 1)


def train_edge_prob_sgd(graph, model, optimizer):
    train_cost, validation_cost = 0, 0
    eps=1e-6
    train_set, validation_set = graph.split(test_size=config.test_size)

    Z = graph.Z
    Z_elems = Z.flatten()
    Z_mean = Z_elems.mean()
    Z = Z.sub(Z_mean)

    for src in train_set:
        optimizer.zero_grad()
        z = Z[src]
        nbrs = graph.nbrs(src)
        if nbrs.shape[0] == 0: continue
        non_nbrs = graph.non_nbrs(src) 
        pos_pair_probs = model(z, Z[nbrs])
        pos_pair_probs = pos_pair_probs.where(
                pos_pair_probs!=0, 
                pos_pair_probs.new_full(pos_pair_probs.shape, eps)
            )
        pos_pair_loss = pos_pair_probs.log().neg().sum()

        neg_pair_probs = model(z, Z[non_nbrs])
        neg_pair_probs = neg_pair_probs.where(neg_pair_probs!=1, neg_pair_probs.new_full(neg_pair_probs.shape, eps))
        neg_pair_loss = (1-neg_pair_probs).log().neg().sum()
        
        loss = pos_pair_loss + neg_pair_loss.div(non_nbrs.shape[0]).mul(nbrs.shape[0])
        
        if torch.isnan(loss): pdb.set_trace()
        loss.backward()
        train_cost += float(loss)
        optimizer.step()

    with torch.no_grad():
        for src in validation_set:
            z = Z[src]
            nbrs = graph.nbrs(src)
            if nbrs.shape[0] == 0: continue
            non_nbrs = graph.non_nbrs(src) 
            pos_pair_probs = model(z, Z[nbrs])
            pos_pair_probs = pos_pair_probs.where(pos_pair_probs!=0, pos_pair_probs.new_full(pos_pair_probs.shape, eps))
            pos_pair_loss = pos_pair_probs.log().neg().sum()

            neg_pair_probs = model(z, Z[non_nbrs])
            neg_pair_probs = neg_pair_probs.where(neg_pair_probs!=1, neg_pair_probs.new_full(neg_pair_probs.shape, eps))
            neg_pair_loss = (1-neg_pair_probs).log().neg().sum()
            
            loss = pos_pair_loss + neg_pair_loss.div(non_nbrs.shape[0]).mul(nbrs.shape[0])
            validation_cost += float(loss) 

    return train_cost, validation_cost


while True:
    context['iteration'] += 1

    # ----------
    #   Model   
    # ----------
    if config.sim_metric == 'edge_prob':
        minimum_validation_cost = inf
        tolerence = config.tolerence_P

        while tolerence != 0:
            model_path = os.path.join(PICKLE_PATH, 'models', '{}_iter_{}'.format(config.model_tag, context['iteration']))
            if os.path.exists(model_path):
                break
            context['n_P'] += 1
            train_cost, validation_cost = train_edge_prob_sgd(
                    graph, 
                    prob_model, 
                    optimizer
                )
            lr_scheduler.step(validation_cost)

            if minimum_validation_cost > validation_cost:
                tolerence = config.tolerence_P
                minimum_validation_cost = validation_cost
                torch.save(prob_model, 
                    os.path.join(PICKLE_PATH, 'models', 
                                 '{}_iter_{}_temp'.format(config.model_tag, context['iteration'])
                                ),
                    )
            else:
                tolerence -= 1
            
            # log stats
            writer.add_scalars('{}/{}'.format(config.model_tag, 'model_cost'),
                    {'train cost': train_cost,
                     'validation cost': validation_cost * 9},
                    context['n_P'] 
                )

        # Load best model
        prob_model = torch.load(
                os.path.join(PICKLE_PATH, 'models', 
                             '{}_iter_{}_temp'.format(config.model_tag, context['iteration'])),
                map_location=device,
            )

        torch.save(prob_model, 
            os.path.join(PICKLE_PATH, 'models', 
                         '{}_iter_{}'.format(config.model_tag, context['iteration'])
                        ),
            )
        sim_metric = prob_model.get_sims

    # ----------
    #   Embed   
    # ----------
        
    minimum_dist = inf
    tolerence = config.tolerence_Z
    
    while tolerence != 0:
        embedding_path = os.path.join(PICKLE_PATH, config.dataset, '{}_iter_{}'.format(config.model_tag, context['iteration']))
        if os.path.exists(embedding_path):
            Z = torch.tensor(pickle.load(open(embedding_path, 'rb'))).to(device)
            graph.Z = Z
            break
        context['n_Z'] += 1
        dist = update_embedding(graph, 
                sim_metric, 
                config.gamma, 
            )
        
        if minimum_dist > dist:
            tolerence = config.tolerence_Z
            minimum_dist = dist
        else:
            tolerence -= 1

        # log stats
        writer.add_scalars('{}/{}'.format(config.model_tag, 'embedding'),
                {'dist': dist},
                context['n_Z'] 
            )

    # Save embedding
    pickle.dump(graph.Z.cpu().numpy(),
            open(os.path.join(PICKLE_PATH, config.dataset, 
                '{}_iter_{}'.format(config.model_tag, context['iteration'])), 'wb'),
        )

    if config.sim_metric == 'cosine': break

