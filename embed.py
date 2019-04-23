import argparse
import logging
import os
import pickle
from time import time

from networkx.classes.function import neighbors
from networkx.classes.function import non_neighbors
from tensorboardX import SummaryWriter
import torch
from torch.distributions import Bernoulli
from torch.nn import LogSoftmax
from torch.nn import NLLLoss
from torch.nn import BCELoss
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from dataset import CoraDataset, CiteseerDataset
from helper import  normalize_elwise
from models import EdgeProbability
from settings import PICKLE_PATH
           

@torch.no_grad()
def update_embedding(network, sim_metric):
    prev_Z = network.Z.clone()
    for src in range(network.Z.shape[0]):
        nbrs = network.A[src].nonzero()
        if len(nbrs) == 0: continue
        sims = sim_metric(network.Z[src], network.Z[nbrs], dim=-1).softmax(0)
        for dst, sim in zip(nbrs, sims):
            network.S[src, dst] = sim

    msg = torch.matmul(network.S, prev_Z)
    network.Z = network.X + config.gamma * msg

    return torch.norm(network.Z - prev_Z, 1)


@torch.no_grad()
def train_edge_prob(network, model):
    start_time = time()
    for src in range(network.Z.shape[0]):
        cost = 0
        z=network.Z[src].unsqueeze(0)

        nbrs = network.A[src].nonzero()
         
        # edge pairs
        pair_probs= model(z, network.Z[nbrs]).unsqueeze(-1).unsqueeze(-1)
        cost += pair_probs.log().neg().sum()
        term = torch.matmul(z.t(), network.Z[nbrs])
        term_A = torch.matmul(term, model.B.weight)
        term_B = torch.matmul(term, model.A.weight)
        
        grad_A = (-(1-pair_probs) * term_A).sum(dim=0)
        grad_B = (-(1-pair_probs) * term_B).sum(dim=0)
        
        # neg pairs
        non_nbrs = (network.A[src]==0).nonzero()
        neg_probs = model(z, network.Z[non_nbrs])
        cost += neg_probs.log().neg().sum()
        sample_index = neg_probs.bernoulli().nonzero()
        for index in DataLoader(sample_index, batch_size=150):
            term = torch.matmul(z.t(), network.Z[index])
            term_A = torch.matmul(term, model.B.weight)
            term_B = torch.matmul(term, model.A.weight)
            
            grad_A.add_((neg_probs[index].unsqueeze(-1) * term_A).sum(0))
            grad_B.add_((neg_probs[index].unsqueeze(-1) * term_B).sum(0))
            
        model.A.weight.add_(grad_A * config.lr)
        model.B.weight.add_(grad_B * config.lr)
        print('{} cost: {:.4f}, time: {:1f}, {:.3f}, {:.3f}'.format(src, cost, time()-start_time, model.A.weight.sum(), model.B.weight.sum()), end='\r')
    return cost

def train_edge_prob_sgd(network, model):
    optimizer = torch.optim.SGD(
            model.parameters(), 
            lr=config.lr,
            )
    cost = 0
    for src in range(network.Z.shape[0]):
        optimizer.zero_grad()
        z = network.Z[src].unsqueeze(0)
        nbrs = network.A[src].nonzero()
        non_nbrs = (network.A[src] == 0).nonzero()

        pos_pair_loss = model(z, network.Z[nbrs]).log().neg().sum()
        neg_pair_loss = (1-model(z, network.Z[non_nbrs])).log().neg().sum()

        loss = pos_pair_loss + neg_pair_loss
        loss.backward()
        cost += float(loss)
        optimizer.step()
    return cost
#            # Update params
#            optimizer = torch.optim.Adam(
#                    edge_prob_model.parameters(), 
#                    lr=config.lr,
#                    )
#
#            for epoch in range(1,config.epoch_P+1):
#                cost = 0
#                loader = DataLoader(list(network.G.nodes()),
#                                    batch_size=config.batch_size,
#                                    shuffle=True)
#
#
#                for batch_idx, batch in enumerate(loader):
#                    optimizer.zero_grad()
#                    indices = [network.index(v) for v in batch]
#                    batch_z = network.Z[indices]
#
#                    # nbrs
#                    batch_nbrs = [list(neighbors(network.G, v)) for v in batch]
#                    number_of_nbrs = [len(nbr) for nbr in batch_nbrs]
#                    batch_nbrs_z = torch.Tensor().to(device)
#                    for batch_nbr in batch_nbrs:
#                        if batch_nbr:
#                            nbr_z = torch.stack([network.z(u) for u in batch_nbr])
#                            while nbr_z.shape[0] < max(number_of_nbrs):
#                                nbr_z = torch.cat([nbr_z, torch.zeros(1, network.d).to(device)])
#                            nbr_z = nbr_z.unsqueeze(0)
#                        else:
#                            nbr_z = torch.zeros(1, max(number_of_nbrs), network.d).to(device)
#
#                        batch_nbrs_z = torch.cat([batch_nbrs_z, nbr_z])
#                    batch_z.requires_grad_(True)
#                    batch_nbrs_z.requires_grad_(True)
#
#                    probs = edge_prob_model(batch_z, batch_nbrs_z)
#                    mask = torch.zeros(probs.shape).to(device)
#                    for i, number in enumerate(number_of_nbrs):
#                        for j in range(number):
#                            mask[i,j] = 1
#                    probs = torch.where(mask>0, probs.log().neg(), torch.tensor([0.]).to(device))
#                    edge_pair_loss = probs.sum()
#                    
#                    # non nbrs
#                    batch_non_nbrs = [list(non_neighbors(network.G, v)) for v in batch]
#                    number_of_non_nbrs = [len(nbr) for nbr in batch_non_nbrs]
#                    batch_non_nbrs_z = torch.Tensor().to(device)
#
#                    for batch_non_nbr in batch_non_nbrs:
#                        indices = [network.index(u) for u in batch_non_nbr]
#                        non_nbr_z = network.Z[indices]
#                        # non_nbr_z = torch.stack([network.G.nodes[u]['z'] for u in batch_non_nbr])
#                        while non_nbr_z.shape[0] < max(number_of_non_nbrs):
#                            non_nbr_z = torch.cat([non_nbr_z, torch.zeros(1, network.d).to(device)])
#                        non_nbr_z = non_nbr_z.unsqueeze(0)
#                        batch_non_nbrs_z = torch.cat([batch_non_nbrs_z, non_nbr_z])
#
#                    batch_non_nbrs_z.requires_grad_(True)
#
#                    probs = edge_prob_model(batch_z, batch_non_nbrs_z)
#                    neg_probs = 1-probs
#                    mask = torch.zeros(probs.shape).to(device)
#                    for i, number in enumerate(number_of_non_nbrs):
#                        for j in range(number):
#                            mask[i,j] = 1
#                    sample_mask = neg_probs.bernoulli()
#                    neg_probs = neg_probs.log().neg()
#                    neg_probs = torch.where(sample_mask * mask > 0, neg_probs, torch.tensor([0.]).to(device))
#                    non_edge_pair_loss = neg_probs.sum()
#
#                    loss = (edge_pair_loss + non_edge_pair_loss) / config.batch_size
#
#                    cost += float(loss)
#                    loss.backward()
#                    optimizer.step()
#                    print('Upd P {}/{} ({:.2f}%) cost: {:.4f}'.format(epoch, config.epoch_P, 100*(batch_idx*config.batch_size + len(batch))/len(network), cost),end='\r')
#                logging.info('Upd P %d/%d cost: %f', epoch, config.epoch_P, cost)
#                writer.add_scalar('{}/cost'.format(config.model_tag), 
#                                  cost, iteration * config.epoch_P + epoch)
#
#            torch.save(edge_prob_model,os.path.join(
#                        PICKLE_PATH, 'models', config.model_tag))

def main(config):
    logging.basicConfig(format='[%(levelname)s] %(message)s',level=logging.DEBUG)
    writer = SummaryWriter(log_dir='runs/{}'.format(config.model_tag))
    
    if torch.cuda.is_available():
        device = torch.device('cuda:{}'.format(config.gpu))
    else:
        device = torch.device('cpu')

    logging.info('DEVICE \t %s', device)
    
    if config.dataset == 'cora':
        network = CoraDataset(sampled=config.sampled, device=device)
        if config.model_load is not None:
            network.load(config)
            
    elif config.dataset == 'citeseer':
        network = CiteseerDataset(sampled=config.sampled, divce=device)
    else:
        raise ValueError
    logging.info('DATASET \t %s', config.dataset)

    if config.sim_metric == 'cosine':
        sim_metric = F.cosine_similarity
    elif config.sim_metric == 'edge_prob':
        if config.model_load is not None:
            edge_prob_model = torch.load(os.path.join(PICKLE_PATH, 'models', 
                        config.model_load))
            checkpoint = torch.load(os.path.join(PICKLE_PATH, 'checkpoints',
                        config.model_load))
            iter_counter_updtP = checkpoint['iter_counter_updtP']

        else:
            edge_prob_model = EdgeProbability(dim=network.d).to(device)
            iter_counter_updtP = 0

        sim_metric = edge_prob_model.get_similarities
    else:
        raise ValueError

    iteration = -1
    while True:
        iteration += 1

        # Optimize Z
        logging.info('ITERATION %d', iteration)
        for epoch in range(1, config.epoch_Z+1):
            distance = update_embedding(network, sim_metric)
            print('optim Z {:d}/{:d} dist: {:.5f}'.format(epoch, config.epoch_Z, distance), end='\r')
            writer.add_scalar('{}/distance'.format(config.model_tag), distance.item(), iteration * config.epoch_Z + epoch)
        logging.info('optim Z %d/%d dist: %f', epoch, config.epoch_Z, distance)
        network.save(config)

        if config.sim_metric == 'edge_prob':
            from time import time
            for epoch in range(1, config.epoch_P+1):
                cost = train_edge_prob_sgd(network, edge_prob_model)
                print('train Pr {:d}/{:d} cost: {:.5f}'.format(epoch, 
                                                               config.epoch_P, 
                                                               cost), end='\r') 
                writer.add_scalar('{}/cost'.format(config.model_tag), cost, iteration * config.epoch_P + epoch)
            logging.info('optim Z %d/%d cost: %f', epoch, config.epoch_Z, cost)
            torch.save(edge_prob_model, os.path.join(PICKLE_PATH, 'models', config.model_tag))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--sampled', action='store_true')

    parser.add_argument('--gamma', type=float, default=0.5)
    parser.add_argument('--epoch_Z', type=int)

    parser.add_argument('--sim_metric', type=str)

    parser.add_argument('--epoch_P', type=int)
     
    parser.add_argument('--lr', type=float, default=0.0001)

    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--model_tag', type=str, default='test')
    parser.add_argument('--model_load', type=str)
    config = parser.parse_args()
    print(config)
    main(config)
