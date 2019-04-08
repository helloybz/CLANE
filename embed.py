import argparse
import os
import pickle

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
            edge_prob_model = EdgeProbability(dim=network.feature_size).to(device)
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
                for v in network.G.nodes():
                    nbrs = list(nx.neighbors(network.G, v))
                    if len(nbrs) == 0: continue
                    
                    z_nbrs = torch.stack([network.G.nodes[u]['z'] for u in nbrs])
                    sims = sim_metric(network.G.nodes[v]['z'], z_nbrs, dim=-1) 
                    sims = F.softmax(sims, dim=-1)
                    sims = sims.squeeze()
                    if sims.dim() == 0: sims = sims.unsqueeze(0)

                    network.G.node[v]['z'] = network.G.nodes[v]['x'] \
                                             + config.gamma * torch.mv(z_nbrs.t(), sims)
                    # TODO: Consider updating embedding including in-edge messages

#                     network.G.node[v]['z'] = network.z(v) / torch.norm(network.z(v), 2)

                distance = torch.norm(network.Z.clone() - previous_Z, 1)
                end_flag = (i == 0 and distance == 0)

                print('Optimize Z | {:4d} | distance: {:10f}'.format(
                            iteration + i, distance), end='\r')
                writer.add_scalar('{}/distance'.format(config.model_tag), 
                        distance.item(), iteration + i)
            network.save(config)
            print('Optimize Z | {:4d} | distance: {:10f}'.format(
                        iteration + i, distance))
       
        if end_flag: break

        if config.sim_metric == 'edge_prob':
            # Update params
            optimizer = torch.optim.Adam(
                    edge_prob_model.parameters(), 
                    lr=config.lr,
                    )
            cost_history = []
            for i in range(1, config.epoch_P + 1):
                cost = 0
    
                for node_idx, v in enumerate(network.G.nodes()):
                    optimizer.zero_grad()
                    edge_prob_model.zero_grad()

                    nbrs = list(nx.neighbors(network.G, v))

                    if len(nbrs) != 0:
                        z_v = network.G.nodes[v]['z'].div(network.G.nodes[v]['z'])
                        z_nbrs = torch.stack([network.G.nodes[u]['z'] for u in nbrs])
                        probs = edge_prob_model(network.G.nodes[v]['z'], z_nbrs)
                        loss_edge = torch.sum(-torch.log(probs))
                    else:
                        loss_edge = 0

                    z_neg_nbrs = torch.stack([network.G.nodes[u]['z'] 
                            for u in nx.non_neighbors(network.G, v)])

                    pos_probs = edge_prob_model(network.G.nodes[v]['z'], z_neg_nbrs)
                    neg_probs = torch.ones(pos_probs.shape[0]).to(device)-pos_probs.squeeze()
                    sampled_indices = neg_probs.detach().bernoulli().byte()
                    neg_probs = neg_probs[sampled_indices]
                    loss_non_edge = torch.sum(-torch.log(neg_probs))
                    if len(nbrs) != 0:
                        loss_non_edge = len(nbrs)*loss_non_edge/len(neg_probs)
                    else:
                        loss_non_edge = 5 * loss_non_edge/len(neg_probs)
                    
                    loss = loss_edge + loss_non_edge
                    cost += loss.data
                    loss.backward()
                    optimizer.step()

                    print('Update Params | {} | Cost: {:4f} | mean: {:4f}'.format(
                                config.epoch_P*iteration+i, cost), end='\r')
                    if node_idx == len(network)-1:
                        print('Update Params | {} | Cost: {:4f}'.format(config.epoch_P*iteration+i, cost))
                
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
    parser.add_argument('--lr', type=float, default=0.0001)

    parser.add_argument('--gpu', type=int, default=0)
    import datetime
    model_tag = str(datetime.datetime.today().isoformat('-')).split('.')[0]
    parser.add_argument('--model_tag', type=str, default=model_tag)
    parser.add_argument('--model_load', type=str)
    config = parser.parse_args()
    print(config)
    main(config)
