import argparse
import os
import pickle

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

        sim_metric = edge_prob_model.get_similarities
    else:
        raise ValueError
    print('[MODEL] LOADED')

    flag_done = False
    
    iter_counter_optZ = 0
    while True:
        # Optimize Z
        print('===================================')
        with torch.no_grad():
            distance_history = []
            while True:
                iter_counter_optZ += 1
                previous_Z = network.Z.clone()
                for v in network.G.nodes():
                    nbrs = neighbors(network.G, v)
                    if len(list(nbrs)) == 0: continue
                    
                    nbrs = torch.stack([network.z(u) for u in neighbors(network.G, v)])
                    sims = sim_metric(network.z(v), nbrs, dim=-1) 
                    sims = F.softmax(sims, dim=0)
                    network.G.node[v]['z'] = network.x(v) \
                                             + config.gamma * torch.mv(nbrs.t(), sims)

                    # TODO: Consider updating embedding including in-edge messages

                    network.G.node[v]['z'] = network.z(v) / torch.norm(network.z(v), 2)

                distance = torch.norm(network.Z.clone() - previous_Z, 2)
                distance_history.append(distance.item())
                flag_done = (distance_history[0]==0)

                print('Optimize Z | {:4d} | distance: {:10f}'.format(
                            iter_counter_optZ, distance), end='\r')
                writer.add_scalar('{}/distance'.format(config.model_tag), 
                        distance.item(), iter_counter_optZ)
                if (len(set(distance_history[-20:])) < 20 and 
                         len(distance_history)>20):
                    if (config.sim_metric == 'cosine' and
                            iter_counter_optZ > len(distance_history)): flag_done=True
                    network.save(config)
                    print('Optimize Z | {:4d} | distance: {:10f}'.format(
                                iter_counter_optZ, distance))
                    break
       
        if flag_done: break

        if config.sim_metric == 'edge_prob':
            # Update params
            optimizer = torch.optim.Adam(
                    edge_prob_model.parameters(), 
                    lr=config.lr,
                    )
            cost_history = []
            epoch = 0
            while True:
                cost = 0
                iter_counter_updtP += 1
                epoch += 1
                for node_idx, v in enumerate(network.G.nodes()):
                    optimizer.zero_grad()
                    edge_prob_model.zero_grad()
                    nbrs = neighbors(network.G, v)
                    nbrs = list(nbrs)
                    if len(nbrs) != 0:
                        nbrs = torch.stack([network.z(u) for u in nbrs])
                        probs = edge_prob_model(network.z(v).unsqueeze(0), nbrs.unsqueeze(0))
                        loss_edge = torch.sum(-torch.log(probs))
                    else:
                        loss_edge = 0

                    neg_nbrs = torch.stack([network.z(u) 
                            for u in list(non_neighbors(network.G, v))])
                    probs = edge_prob_model(network.z(v).unsqueeze(0), neg_nbrs.unsqueeze(0))
                    neg_probs = 1-probs
                    neg_probs = neg_probs[(1-probs).clone().detach().bernoulli().byte()]
                    loss_non_edge = torch.sum(-torch.log(neg_probs))

                    # Backprop & update
                    loss = loss_edge + loss_non_edge
                    cost += loss.data
                    loss.backward()
                    optimizer.step()
                    print('Update Params | {} | Cost: {:4f}'.format(epoch, cost), end='\r')
                    if node_idx == len(network)-1:
                        print('Update Params | {} | Cost: {:4f}'.format(epoch, cost))
                
                cost_history.append(int(cost))
                
                writer.add_scalar('{}/cost'.format(config.model_tag), 
                                  cost, iter_counter_updtP)

                if (len(set(cost_history[-20:])) < 10 and
                        len(cost_history) > 20):
                    break

            torch.save(edge_prob_model,os.path.join(
                        PICKLE_PATH, 'models', config.model_tag))
            checkpoint = {
                'iter_counter_updtP': iter_counter_updtP
            }
            torch.save(checkpoint, os.path.join(
                        PICKLE_PATH, 'checkpoints', config.model_tag))

        network.save(config)

    pickle.dump(network.Z.cpu().data.numpy(),
                open(os.path.join(PICKLE_PATH, 
                                  config.dataset,
                                  config.model_tag), 'wb'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--sampled', action='store_true')

    parser.add_argument('--gamma', type=float, default=0.9)

    parser.add_argument('--sim_metric', type=str)
    
    parser.add_argument('--lr', type=float, default=0.0001)

    parser.add_argument('--gpu', type=int, default=0)
    import datetime
    model_tag = str(datetime.datetime.today().isoformat('-')).split('.')[0]
    parser.add_argument('--model_tag', type=str, default=model_tag)
    parser.add_argument('--model_load', type=str)
    config = parser.parse_args()
    print(config)
    main(config)
