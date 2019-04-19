import argparse
import logging
import os
import pickle
import pdb

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

        sim_metric = edge_prob_model.forward
    else:
        raise ValueError

    iteration = -1
    while True:
        iteration += 1

        # Optimize Z
        logging.info('ITERATION %d', iteration)
        with torch.no_grad():
            for epoch in range(1, config.epoch_Z+1):
                prev_Z = network.Z.clone()
                for v in network.G.nodes():
                    nbrs = list(neighbors(network.G, v))
                    if len(nbrs) == 0: continue
                    
                    nbrs_z = torch.stack([prev_Z[network.index(u)] for u in nbrs])
                    sims = sim_metric(prev_Z[network.index(v)], nbrs_z, dim=-1) 
                    sims = F.softmax(sims, dim=0)
                    network.G.node[v]['z'] = network.G.nodes[v]['x'] \
                                             + config.gamma * torch.mv(nbrs_z.t(), sims)

                distance = torch.norm(network.Z - prev_Z, 1)
                print('optim Z {:d}/{:d} dist: {:.5f}'.format(epoch, config.epoch_Z, distance), end='\r')
                writer.add_scalar('{}/distance'.format(config.model_tag), distance.item(), iteration * config.epoch_Z + epoch)
                
                if epoch % 100 == 0: network.save(config)
                   
            logging.info('optim Z %d/%d dist: %f', epoch, config.epoch_Z, distance)

        if config.sim_metric == 'edge_prob':
            # Update params
            optimizer = torch.optim.Adam(
                    edge_prob_model.parameters(), 
                    lr=config.lr,
                    )

            for epoch in range(1,config.epoch_P+1):
                cost = 0
                loader = DataLoader(list(network.G.nodes()),
                                    batch_size=config.batch_size,
                                    shuffle=True)
                for batch_idx, batch in enumerate(loader):
                    optimizer.zero_grad()
                    indices = [network.index(v) for v in batch]
                    batch_z = network.Z[indices]

                    # nbrs
                    batch_nbrs = [list(neighbors(network.G, v)) for v in batch]
                    number_of_nbrs = [len(nbr) for nbr in batch_nbrs]
                    batch_nbrs_z = torch.Tensor().to(device)
                    for batch_nbr in batch_nbrs:
                        if batch_nbr:
                            nbr_z = torch.stack([network.z(u) for u in batch_nbr])
                            while nbr_z.shape[0] < max(number_of_nbrs):
                                nbr_z = torch.cat([nbr_z, torch.zeros(1, network.d).to(device)])
                            nbr_z = nbr_z.unsqueeze(0)
                        else:
                            nbr_z = torch.zeros(1, max(number_of_nbrs), network.d).to(device)

                        batch_nbrs_z = torch.cat([batch_nbrs_z, nbr_z])
                    batch_z.requires_grad_(True)
                    batch_nbrs_z.requires_grad_(True)

                    probs = edge_prob_model(batch_z, batch_nbrs_z)
                    mask = torch.zeros(probs.shape).to(device)
                    for i, number in enumerate(number_of_nbrs):
                        for j in range(number):
                            mask[i,j] = 1
                    probs = torch.where(mask>0, probs.log().neg(), torch.tensor([0.]).to(device))
                    edge_pair_loss = probs.sum()
                    
                    # non nbrs
                    batch_non_nbrs = [list(non_neighbors(network.G, v)) for v in batch]
                    number_of_non_nbrs = [len(nbr) for nbr in batch_non_nbrs]
                    batch_non_nbrs_z = torch.Tensor().to(device)

                    for batch_non_nbr in batch_non_nbrs:
                        indices = [network.index(u) for u in batch_non_nbr]
                        non_nbr_z = network.Z[indices]
                        # non_nbr_z = torch.stack([network.G.nodes[u]['z'] for u in batch_non_nbr])
                        while non_nbr_z.shape[0] < max(number_of_non_nbrs):
                            non_nbr_z = torch.cat([non_nbr_z, torch.zeros(1, network.d).to(device)])
                        non_nbr_z = non_nbr_z.unsqueeze(0)
                        batch_non_nbrs_z = torch.cat([batch_non_nbrs_z, non_nbr_z])

                    batch_non_nbrs_z.requires_grad_(True)

                    probs = edge_prob_model(batch_z, batch_non_nbrs_z)
                    neg_probs = 1-probs
                    mask = torch.zeros(probs.shape).to(device)
                    for i, number in enumerate(number_of_non_nbrs):
                        for j in range(number):
                            mask[i,j] = 1
                    sample_mask = neg_probs.bernoulli()
                    neg_probs = neg_probs.log().neg()
                    neg_probs = torch.where(sample_mask * mask > 0, neg_probs, torch.tensor([0.]).to(device))
                    non_edge_pair_loss = neg_probs.sum()

                    loss = (edge_pair_loss + non_edge_pair_loss) / config.batch_size

                    cost += float(loss)
                    loss.backward()
                    optimizer.step()
                    print('Upd P {}/{} ({:.2f}%) cost: {:.4f}'.format(epoch, config.epoch_P, 100*(batch_idx*config.batch_size + len(batch))/len(network), cost),end='\r')
                logging.info('Upd P %d/%d cost: %f', epoch, config.epoch_P, cost)
                writer.add_scalar('{}/cost'.format(config.model_tag), 
                                  cost, iteration * config.epoch_P + epoch)

            torch.save(edge_prob_model,os.path.join(
                        PICKLE_PATH, 'models', config.model_tag))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--sampled', action='store_true')

    parser.add_argument('--gamma', type=float, default=0.5)
    parser.add_argument('--epoch_Z', type=int)

    parser.add_argument('--sim_metric', type=str)

    parser.add_argument('--epoch_P', type=int)
    parser.add_argument('--batch_size', type=int)
     
    parser.add_argument('--lr', type=float, default=0.0001)

    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--model_tag', type=str, default='test')
    parser.add_argument('--model_load', type=str)
    config = parser.parse_args()
    print(config)
    main(config)
