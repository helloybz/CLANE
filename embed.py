import argparse
import os
import pdb
import pickle
import time

import numpy as np
from numpy.random import choice
from networkx.classes.function import neighbors
from networkx.classes.function import non_neighbors
from networkx.linalg.attrmatrix import attr_matrix
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
           

def edge_prob_method(dataset, **kwargs):
    writer = SummaryWriter(log_dir='runs/{}'.format(config.model_tag))
    
    if config.model_load is not None:
        edge_prob_model = torch.load(os.path.join(PICKLE_PATH, 'models', 
                    config.model_load))
        checkpoint = torch.load(os.path.join(PICKLE_PATH, 'checkpoints',
                    config.model_load))
        iter_counter = checkpoint['iter_counter']
        dataset.Z = checkpoint['dataset_Z']

    else:
        edge_prob_model = EdgeProbability(dim=dataset.X.shape[1])
        iter_counter = 0
        dataset.Z = dataset.X.clone()
    
    edge_prob_model = edge_prob_model.cuda(kwargs['device'])
    dataset.Z = dataset.Z.to(kwargs['device'])
    
    while True:
        iter_counter += 1

        ### Optimize Embedding. ###

        step1_converged = False
        step1_counter = 1
        step1_start = time.time()
        delta_list = []

        with torch.no_grad():
            while not step1_converged:
                maximum_delta = -np.inf
                previous_Z = dataset.Z.clone()
                
                for doc_id in range(len(dataset)):
                    z = dataset[doc_id]
                    ref_indices = dataset.A[doc_id].nonzero()
                    Z_ref = dataset[ref_indices]
                    if Z_ref.dim() == 1: Z_ref.unsqueeze_(0)
                    
                    messages = torch.mv(Z_ref.squeeze(-2).t(), 
                            edge_prob_model.get_similarities(z, Z_ref))
                    newly_calculated_z = dataset.X[doc_id] \
                                         + (config.gamma * messages)

                    delta = torch.norm(newly_calculated_z - z, 2)
                    maximum_delta = max(maximum_delta, delta)

                    dataset.Z[doc_id] = newly_calculated_z

                    print('UpdateZ / {0} / maximum_distance {1:.6f} /' + \
                            '{2:4d}/{3:4d}'.format(step1_counter, 
                                                  maximum_delta,
                                                  doc_id,
                                                  len(dataset)), end='\r')
                    
                delta_list.append(maximum_delta)
                
                if len(delta_list) == 11:
                    delta_list = delta_list[1:]
                
                if maximum_delta < config.epsilon:
                    step1_converged = True
                elif (len(delta_list) == 10 and
                        max(delta_list) - min(delta_list) < config.epsilon):
                    step1_converged = True
                else:
                    # still not converged
                    step1_counter += 1
                
            print('Step 1 done')
            step1_end = time.time()
            step1_time = torch.Tensor([step1_end-step1_start])
            step1_counter_ = torch.Tensor([step1_counter])
            writer.add_scalar('{}/Convergence time Updating Z' \
                    .format(config.model_tag), step1_time[0], iter_counter)
            print("Z convergence time: {:.3f}" \
                    .format(step1_end - step1_start))

        # check if Z is updated
        delta_Z = torch.norm(dataset.Z - previous_Z, 2)
        if delta_Z == 0:
            print('Z is not updated.')
            break
        else:
            print('Z is updated. Go to step2. {}'.format(delta_Z))
            pickle.dump(dataset.Z.clone().cpu().data.numpy(), 
                        open(os.path.join(PICKLE_PATH, 
                                          config.dataset,
                                          'temp_Z_{}'.format(
                                              config.model_tag)), 'wb'))
         
        # step 2, "Update W"
        step2_start = time.time()
        for epoch in range(config.epoch):
            step2_epoch_start = time.time()
            cost = 0
            edge_pairs = dataset.A.clone().nonzero()
            nonedge_pairs = (dataset.A==0).nonzero()
            optimizer = torch.optim.Adam(
                    edge_prob_model.parameters(), 
                    lr=config.lr,
                    )
            
            for idx, pair in enumerate(DataLoader(edge_pairs, 
                        batch_size=config.batch_size)):
                Z_ = dataset[pair].detach().requires_grad_()
                
                edge_probs = edge_prob_model.forward1(Z_).squeeze()
                output = torch.sum(-torch.log(edge_probs))
                
                edge_prob_model.zero_grad()
                optimizer.zero_grad()
                output.backward()
                optimizer.step()
                cost = cost + output.item() 
                
                print("Update Params for connected pair: {:4d}/{:4d}".format(
                        idx*config.batch_size+len(pair), len(edge_pairs)), end='\r')

            for idx, pair in enumerate(DataLoader(nonedge_pairs, 
                        batch_size=config.batch_size)):
                Z_ = dataset[pair].detach().requires_grad_()
                edge_probs = edge_prob_model.forward1(Z_).squeeze()
                negative_edge_probs = 1-edge_probs
                output = torch.sum(-torch.log(negative_edge_probs))
                
                edge_prob_model.zero_grad()
                optimizer.zero_grad()
                output.backward()
                optimizer.step()
                cost = cost + output.item()

                print("Update Params for not connected pair: {:4d}/{:4d}".format(
                        idx*config.batch_size+len(pair), len(nonedge_pairs)), end='\r')

            # 일정 에폭마다 샘플된 페어들에 대해서 코스트 계산
            if epoch % config.log_period == 0:
                print('epoch: {} , cost: {}'.format(epoch, -cost))
                writer.add_scalar('{}/Cost'.format(config.model_tag),
                                  cost, ((iter_counter-1) * config.epoch) + epoch)
            step2_epoch_end = time.time()
            step2_epoch_time = torch.Tensor([step2_epoch_end-step2_epoch_start])
            print("step2 epoch time: {:.3f}".format(step2_epoch_end - step2_epoch_start))
        print('step2 done')
        step2_end = time.time()
        print("step2 time: {:.3f}".format(step2_end - step2_start)
        )
        # model save
        torch.save(edge_prob_model,os.path.join(
                    PICKLE_PATH, 'models', config.model_tag))
        checkpoint = {
            'iter_counter': iter_counter,
            'dataset_Z': dataset.Z.clone()
        }
        torch.save(checkpoint, os.path.join(
                    PICKLE_PATH, 'checkpoints', config.model_tag))
    writer.close()
    return dataset


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
        iter_counter = 0
    elif config.sim_metric == 'edge_prob':
        if config.model_load is not None:
            edge_prob_model = torch.load(os.path.join(PICKLE_PATH, 'models', 
                        config.model_load))
            checkpoint = torch.load(os.path.join(PICKLE_PATH, 'checkpoints',
                        config.model_load))
            iter_counter = checkpoint['iter_counter']-1
        else:
            edge_prob_model = EdgeProbability(dim=network.feature_size).to(device)
            iter_counter = 0

        sim_metric = edge_prob_model.get_similarities
    else:
        raise ValueError
    print('[MODEL] LOADED')

    flag_done = False
    while True:
        iter_counter += 1
        # Optimize Z
        print('===================================')
        with torch.no_grad():
            iter_counter_optZ = 0
            distance_history = []
            while True:
                iter_counter_optZ += 1
                previous_Z = network.Z.clone()
                for v in network.G.nodes():
                    nbrs = neighbors(network.G, v)
                    if len(list(nbrs)) == 0:
                        continue
                    nbrs = torch.stack([network.z(u) for u in neighbors(network.G, v)])
                    sims = sim_metric(network.z(v), nbrs, dim=-1) 
                    sims = F.softmax(sims, dim=0)
                    network.G.node[v]['z'] = network.x(v) \
                                             + config.gamma * torch.mv(nbrs.t(), sims)
                    network.G.node[v]['z'] = network.z(v) / torch.norm(network.z(v), 2)

                distance = torch.norm(network.Z.clone() - previous_Z, 2)
                distance_history.append(distance.item())
                flag_done = (distance==0)
                print('Optimize Z | {:4d} | distance: {:10f}'.format(
                            iter_counter_optZ, distance), end='\r')
                if (distance < config.epsilon or
                        (len(set(distance_history[-10:])) == 1 and 
                         len(distance_history)>10)):
                    writer.add_scalar('{}/distance'.format(config.model_tag), 
                            distance.item(), iter_counter)
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
            for epoch in range(config.epoch):
                cost = 0
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
                    print('Update Params | {} | Cost: {}'.format(epoch, cost), end='\r')
                    if node_idx == len(network)-1:
                        print('Update Params | {} | Cost: {}'.format(epoch, cost))

                writer.add_scalar('{}/cost'.format(config.model_tag), 
                                  cost, 
                                  iter_counter*config.epoch + epoch)
        

            torch.save(edge_prob_model,os.path.join(
                        PICKLE_PATH, 'models', config.model_tag))
            checkpoint = {
                'iter_counter': iter_counter,
            }
            torch.save(checkpoint, os.path.join(
                        PICKLE_PATH, 'checkpoints', config.model_tag))

        network.save(config)
    pickle.dump(network.Z.cpu().data.numpy(), 
                open(os.path.join(PICKLE_PATH, 
                                  config.dataset,
                                  'Z_{}_{}_gamma{}_thr{}'.format(
                                     config.dataset,
                                     config.sim_metric,
                                     config.gamma,
                                     config.epsilon)), 
                     'wb'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--sampled', action='store_true')

    parser.add_argument('--gamma', type=float, default=0.9)
    parser.add_argument('--epsilon', type=float, default=0.0001)

    parser.add_argument('--sim_metric', type=str)
    
    parser.add_argument('--epoch', type=int, default=1)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--neg_smpl_size', type=int, default=5)
    parser.add_argument('--log_period', type=int, default=1)

    parser.add_argument('--gpu', type=int, default=0)
    import datetime
    model_tag = str(datetime.datetime.today().isoformat('-')).split('.')[0]
    parser.add_argument('--model_tag', type=str, default=model_tag)
    parser.add_argument('--model_load', type=str)
    config = parser.parse_args()
    print(config)
    main(config)
