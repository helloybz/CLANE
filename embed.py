import argparse
import os
import pickle
import pdb

import networkx as nx
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
    writer = SummaryWriter(log_dir='runs/{}'.format(config.model_tag))
    
    if torch.cuda.is_available():
        device = torch.device('cuda:{}'.format(config.gpu))
    else:
        device = torch.device('cpu')
    print('[DEVICE] {}'.format(device))
    
    print('[DATASET] LOADING, {}'.format(config.dataset))
    if config.dataset == 'cora':
        network = CoraDataset(sampled=config.sampled, device=device)
        if config.model_load is not None:
            network.load(config)
            
    elif config.dataset == 'citeseer':
        network = CiteseerDataset(sampled=config.sampled, divce=device)
    else:
        raise ValueError
    print('[DATASET] LOADED, {}'.format(config.dataset))

    print('[MODEL] LOADING')
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
    print('[MODEL] LOADED')

    end_flag = False
    
    iteration = -1
    while True:
        iteration += 1
        
        # Optimize Z
        print('===================================')
        with torch.no_grad():
            for i in range(config.epoch_Z):
                previous_Z = network.Z.clone()
                nodes = list(network.G.nodes())
                for v in network.G.nodes():
                    nbrs = list(nx.neighbors(network.G, v))
                    if len(nbrs) == 0: continue
                    
                    z_nbrs = torch.stack([previous_Z[nodes.index(u)] for u in nbrs])
                    sims = sim_metric(previous_Z[nodes.index(v)], z_nbrs, dim=-1) 
                    sims = F.softmax(sims, dim=-1)
                    sims = sims.squeeze()
                    if sims.dim() == 0: sims = sims.unsqueeze(0)

                    network.G.node[v]['z'] = network.G.nodes[v]['x'] \
                                             + config.gamma * torch.mv(z_nbrs.t(), sims)

                distance = torch.norm(network.Z.clone() - previous_Z, 1)
                end_flag = (i == 0 and distance == 0)
                
                print('Optimize Z | {:4d} | distance: {:10f} | Cumm distance: '.format(iteration + i, distance), end='\r')
                writer.add_scalar('{}/distance'.format(config.model_tag), 
                        distance.item(), config.epoch_Z*iteration + i)
            network.save(config)
            print('Optimize Z | {:4d} | distance: {:10f} | Cumm distance: '.format(iteration + i, distance))
       
        if end_flag: break

        if config.sim_metric == 'edge_prob':
            # Update params
            optimizer = torch.optim.Adam(
                    edge_prob_model.parameters(), 
                    lr=config.lr)
            for epoch in range(1, config.epoch_P + 1):
                cost = 0
                loader = DataLoader(list(network.G.nodes()), 
                                    batch_size=config.batch_size, 
                                    shuffle=True)
                for batch_idx, batch in enumerate(loader):
                    optimizer.zero_grad()
                   
                    batch_z = torch.stack([network.G.nodes[v]['z'] for v in batch])

                    batch_nbrs = [list(nx.neighbors(network.G, v)) for v in batch]
                    number_of_nbrs = [len(nbr) for nbr in batch_nbrs]
                    batch_nbrs_z = torch.Tensor().to(device)
                    for batch_nbr in batch_nbrs:
                        if batch_nbr:
                            nbr_z = torch.stack([network.z(u) for u in batch_nbr])
                            while nbr_z.shape[0] < max(number_of_nbrs):
                                nbr_z = torch.cat([nbr_z, 
                                                   torch.zeros(1,network.d).to(device)])
                            nbr_z = nbr_z.unsqueeze(0)
                        else:
                            nbr_z = torch.zeros(1, max(number_of_nbrs), network.d).to(device)
                        batch_nbrs_z = torch.cat([batch_nbrs_z, nbr_z])

                    batch_z.requires_grad_(True)
                    batch_nbrs_z.requires_grad_(True)

                    probs = edge_prob_model(batch_z, batch_nbrs_z)
                    mask = torch.zeros(probs.shape).to(device)
                    for i,number in enumerate(number_of_nbrs):
                        for j in range(number):
                            mask[i,j] = 1
                    edge_pair_mean = torch.where(mask > 0, probs, torch.tensor([0.]).to(device)).sum()/sum(number_of_nbrs)
                    probs = torch.where(mask > 0, probs.log().neg(), torch.tensor([0.]).to(device))
                    edge_pair_loss = torch.sum(probs)

                    batch_non_nbrs = [list(nx.non_neighbors(network.G, v)) for v in batch]
                    number_of_non_nbrs = [len(nbr) for nbr in batch_non_nbrs]
                    batch_non_nbrs_z = torch.Tensor().to(device)
                    
                    for batch_non_nbr in batch_non_nbrs:
                        non_nbr_z = torch.stack([network.z(u) for u in batch_non_nbr])
                        while non_nbr_z.shape[0] < max(number_of_non_nbrs):
                            non_nbr_z = torch.cat([non_nbr_z,
                                                   torch.zeros(1, network.d).to(device)])
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
                    non_edge_pair_mean = torch.where(mask > 0, neg_probs, torch.tensor([0.]).to(device)).sum()/sum(number_of_non_nbrs)
                    neg_probs = neg_probs.log().neg()
                    neg_probs = torch.where(sample_mask * mask > 0, neg_probs, torch.tensor([0.]).to(device))
                    non_edge_pair_loss = torch.sum(neg_probs)
                    loss = (edge_pair_loss + non_edge_pair_loss)/config.batch_size
                    cost += float(loss)
                    loss.backward()
                    optimizer.step()

                    print('Update Params | ' + \
#                            'iter: {} | '.format(iteration) + \
#                            'epc: {} | '.format(epoch) + \
#                            'btc_idx: {} | '.format(batch_idx) + \
                            'cost: {:3f} | '.format(cost) + \
                            'pos_prob: {:2f} | '.format(edge_pair_mean) + \
                            'neg__prob: {:2f} | '.format(non_edge_pair_mean), end='\r')
                

                print('Update Params | ' + \
                        'Iter: {:5d} | '.format(iteration) + \
                        'Epoch: {:3d} | '.format(epoch) + \
                        'Cost: {:5f} | '.format(cost))
               
                writer.add_scalar('{}/cost'.format(config.model_tag), 
                              cost, config.epoch_P*iteration + epoch)

#                for node_idx, v in enumerate(network.G.nodes()):
#                    optimizer.zero_grad()
#                    edge_prob_model.zero_grad()
#
#                    nbrs = list(nx.neighbors(network.G, v))
#                    if len(nbrs) != 0:
#                        z_v = network.G.nodes[v]['z'].div(network.G.nodes[v]['z'])
#                        z_nbrs = torch.stack([network.G.nodes[u]['z'] for u in nbrs])
#                        probs = edge_prob_model(network.G.nodes[v]['z'], z_nbrs)
#                        loss_edge = torch.sum(-torch.log(probs))
#                    else:
#                        loss_edge = 0
#
#                    z_neg_nbrs = torch.stack([network.G.nodes[u]['z'] 
#                            for u in nx.non_neighbors(network.G, v)])
#
#                    pos_probs = edge_prob_model(network.G.nodes[v]['z'], z_neg_nbrs)
#                    if (pos_probs < 0).sum() != 0 or (pos_probs > 1).sum() != 0 or (torch.isnan(pos_probs).sum() != 0):
#                        pdb.set_trace()
#                        
#                    neg_probs = torch.ones(pos_probs.shape[0]).to(device)-pos_probs.squeeze()
#                    sampled_indices = neg_probs.detach().bernoulli().byte()
#                    neg_probs = neg_probs[sampled_indices]
#                    loss_non_edge = torch.sum(-torch.log(neg_probs))
#                    if len(nbrs) != 0:
#                        loss_non_edge = len(nbrs)*loss_non_edge/len(neg_probs)
#                    else:
#                        loss_non_edge = 5*loss_non_edge/len(neg_probs)
#                    
#                    loss = loss_edge + loss_non_edge
#                    cost += loss.data
#                    loss.backward()
#                    optimizer.step()
#
#                    print('Update Params | {} | Cost: {:4f} | Param Sum: {:4f}, {:4f} {:4f} {:4f}'.format(
#                                config.epoch_P*iteration+i, cost,
#                                edge_prob_model.A.weight.sum(),
#                                edge_prob_model.B.weight.sum(),probs.mean(), neg_probs.mean()), end='\r')
#                    if node_idx == len(network)-1:
#                        print('Update Params | {} | Cost: {:4f} | Param Sum: {:4f}, {:4f} {:4f} {:4f}'.format(
#                                config.epoch_P*iteration+i, cost,
#                                edge_prob_model.A.weight.sum(),
#                                edge_prob_model.B.weight.sum(),probs.mean(), neg_probs.mean()))
                
                writer.add_scalar('{}/cost'.format(config.model_tag), 
                                  cost, config.epoch_P*iteration + i)

            torch.save(edge_prob_model,os.path.join(
                        PICKLE_PATH, 'models', config.model_tag))
            checkpoint = {
            }
            torch.save(checkpoint, os.path.join(
                        PICKLE_PATH, 'checkpoints', config.model_tag))

    pickle.dump(network.Z.cpu().data.numpy(),
                open(os.path.join(PICKLE_PATH, 
                                  config.dataset,
                                  config.model_tag), 'wb'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--sampled', action='store_true')

    parser.add_argument('--gamma', type=float, default=0.9)
    parser.add_argument('--epoch_Z', type=int, default=100)

    parser.add_argument('--sim_metric', type=str)
    parser.add_argument('--epoch_P', type=int, default=300)
    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument('--lr', type=float, default=0.0001)

    parser.add_argument('--gpu', type=int, default=0)
    import datetime
    model_tag = str(datetime.datetime.today().isoformat('-')).split('.')[0]
    parser.add_argument('--model_tag', type=str, default=model_tag)
    parser.add_argument('--model_load', type=str)
    config = parser.parse_args()
    print(config)
    main(config)
