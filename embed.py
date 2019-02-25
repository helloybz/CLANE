import argparse
import os
import pdb
import pickle
import time

import numpy as np
import torch
from scipy.spatial.distance import euclidean
from torch.distributions import Bernoulli
from torch.nn import CosineSimilarity
from torch.utils.data import DataLoader, Dataset

from dataset import CoraDataset, CiteseerDataset
from models import EdgeProbability
from settings import PICKLE_PATH


def cosine_sim_method(dataset, **kwargs):
    is_converged = False
    cosine_sim = CosineSimilarity(dim=-1)
    iter_counter = 0
    dataset.Z = dataset.X.clone()

    while not is_converged:
        iter_counter += 1
        maximum_delta = -np.inf
        
        for doc_id in range(len(dataset)):
            z = dataset[doc_id]
            ref_indices = dataset.A[doc_id].nonzero().squeeze()
            Z_ref = dataset[ref_indices]
            if Z_ref.dim() == 1 : Z_ref.unsqueeze_(0)
            
            messages = torch.mv(Z_ref.t(), 
                                torch.nn.functional.softmax(
                                    (1+cosine_sim(z, Z_ref))/2,
                                    dim=0))
            newly_calculated_z = dataset.X[doc_id] + config.gamma * messages
            delta = torch.norm(newly_calculated_z - z, 2)
            if delta > 100: pdb.set_trace()
            maximum_delta = max(maximum_delta, delta)
            dataset.Z[doc_id] = newly_calculated_z
            
            print('UpdateZ / {0} / maximum_distance {1:.6f} / {2:4d}/{3:4d}'\
                  .format(iter_counter, 
                          maximum_delta, 
                          doc_id, len(dataset)),
                  end='\r')

        if maximum_delta < config.epsilon:
            is_converged = True
    
    return dataset
           

def edge_prob_method(dataset, **kwargs):
    torch.set_grad_enabled(False)
   
    edge_prob_model = EdgeProbability(dim=dataset.X.shape[1])
    edge_prob_model = edge_prob_model.cuda(kwargs['device'])
    for p in edge_prob_model.parameters(): p.requires_grad = False
    
    dataset.Z = dataset.X.clone()
    
    while True:
        step1_converged = False

        step1_counter = 1
        distance_list = torch.Tensor([]).cuda(kwargs['device'])

        step1_start = time.time()
        while not step1_converged:
            max_distance = -np.inf

            for doc_id in range(len(dataset)):
                z1 = dataset[doc_id]
                z2s = dataset[dataset.A[doc_id].nonzero()]
                
                similarities = edge_prob_model.get_similarities(z1, z2s)

                message = config.gamma * torch.mv(torch.t(z2s.squeeze(-2)),
                                                  similarities)

                newly_calculated_z = dataset.X[doc_id] + message

                distance = dataset.Z[doc_id] - newly_calculated_z
                distance = torch.sqrt(torch.sum(distance ** 2)).unsqueeze(-1)

                max_distance = max(max_distance, distance)

                dataset.Z[doc_id] = newly_calculated_z

                print('UpdateZ / {0} / maximum_distance {1:.6f} / {2:4d}/{3:4d}'\
                      .format(step1_counter, 
                              max_distance.item(), 
                              doc_id,
                              dataset.Z.shape[0]),
                      end='\r')

                 # UpdateZ done
            
            distance_list = torch.cat((distance_list, max_distance))

            if len(distance_list) == 10:
                distance_list = distance_list[1:]

            if max_distance < config.threshold_distance:
                step1_converged = True
            elif (len(distance_list) == 10 and
                    max(distance_list) - min(distance_list) <
                    config.threshold_distance):
                step1_converged = True
            else:
                # still not converged
                step1_counter += 1

        print('Step 1 done')
        step1_end = time.time()
        print("Z convergence time: {:.3f}".format(step1_end - step1_start))


        # check if Z is updated
        if torch.max(dataset.Z - previous_Z) < 0.01:
            print('Z is not updated.')
            break
        else:
            print('Z is updated. Go to step2.')

        # step 2, "Update W"
        for epoch in range(config.epoch):
            step2_epoch_start = time.time()
            
            J_A = torch.zeros(edge_prob_model.A.weight.shape) \
                       .to(kwargs['device'])
            J_B = torch.zeros(edge_prob_model.B.weight.shape) \
                       .to(kwargs['device'])
            
            if epoch % config.log_period == 0:
                cost = torch.zeros([1]).to(kwargs['device'])
             
            for z1, z2 in dataset.get_all_edges():
                prob_edge = edge_prob_model(dataset[z1], dataset[z2])
                
                is_selected = (1-prob_edge).bernoulli()
                
                if is_selected == 1:
                    
                    j_A = torch.mm(dataset[z2].unsqueeze(-1),
                                   dataset[z1].unsqueeze(0))
                    j_A = torch.mm(edge_prob_model.B.weight.clone(),
                                   j_A)
                    
                    j_B = torch.mm(dataset[z2].unsqueeze(-1),
                                   dataset[z1].unsqueeze(0))
                    j_B = torch.mm(j_B,
                                   torch.t(edge_prob_model.A.weight.clone()))
                    
                    J_A.add_(j_A)
                    J_B.add_(j_B)

                    if epoch % config.log_period == 0:
                        cost.sub_(torch.log(prob_edge))                    
                
                print("Update Params edge pair: {:4d}/{:4d}".format(z1,len(dataset)), end='\r') 
             
            for z1, z3s in dataset.get_all_non_edges():
                print("Update Params non-edge pair: {:4d}/{:4d}".format(z1, len(dataset)), end='\r')
                
                prob_edges = edge_prob_model.forward_batch(
                                 dataset[z1].unsqueeze(0),
                                 dataset[z3s])
                
                if config.sampling == 'bernoulli':
                    is_selected = prob_edges.bernoulli().squeeze()
                elif config.sampling == 'top100':
                    is_selected = prob_edges.sort(dim=0)[1][:100].squeeze()
                else:
                    raise ValueError

                z1 = dataset[z1]
                z3s = dataset[z3s.squeeze()][is_selected]

                if epoch % config.log_period == 0:
                    pdb.set_trace()
                    cost.sub_(torch.sum(torch.log(prob_edges[is_selected])))
                  
                for indices in DataLoader(range(z3s.shape[0]),
                                          batch_size=400):
                    
                    j_A = torch.matmul(z1.unsqueeze(-1),
                                       z3s[indices].unsqueeze(-2))
                    j_A = torch.matmul(edge_prob_model.B.weight,
                                       j_A)
                    j_A = torch.sum(j_A, dim=0)
                    J_A.sub_(j_A)
                    # del j_A; torch.cuda.empty_cache()

                    j_B = torch.matmul(z3s[indices].unsqueeze(-1),
                                       z1.unsqueeze(0))
                    j_B = torch.transpose(j_B, 1, 2)
                    j_B = edge_prob_model.A(j_B)
                    j_B = torch.transpose(j_B, 1, 2)
                    j_B = torch.sum(j_B, dim=0)
                    J_B.sub_(j_B)
                    # del j_B; torch.cuda.empty_cache()

            edge_prob_model.A.weight.data.add_(config.lr * J_A)
            edge_prob_model.B.weight.data.add_(config.lr * J_B)

            # 일정 에폭마다 샘플된 페어들에 대해서 코스트 계산
            if epoch % config.log_period == 0:
                print('epoch: {} , cost: {}'.format(epoch, cost))
                
            step2_epoch_end = time.time()
            print("step2 epoch time: {:.3f}".format(step2_epoch_end - step2_epoch_start))
        print('step2 done')
        step2_end = time.time()
        print("step2 time: {:.3f}".format(step2_end - step2_start)
        )
    return dataset


def main(config):
    if torch.cuda.is_available():
        device = torch.device('cuda:{}'.format(config.gpu))
    else:
        device = torch.device('cpu')
    print('[DEVICE] {}'.format(device))

    if config.dataset == 'cora':
        dataset = CoraDataset(device=device)
    elif config.dataset == 'citeseer':
        dataset = CiteseerDataset(device=device)
    else:
        raise ValueError
    print('DATASET LOADED.')

    if config.method == 1:
        dataset = cosine_sim_method(dataset, device=device)
    elif config.method == 2:
        dataset = edge_prob_method(dataset, device=device)
    else:
        raise ValueError 

    pickle.dump(dataset.Z.cpu().data.numpy(), 
                open(os.path.join(PICKLE_PATH, 
                                  config.dataset,
                                  'v_{}_method{}_epoch{}_batch{}_gamma{}_thr{}'.format(
                                     config.dataset,
                                     config.method,
                                     config.epoch,
                                     config.batch_size,
                                     config.gamma,
                                     config.epsilon)), 
                     'wb'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--method', type=int, default=1)
    parser.add_argument('--gamma', type=float, default=0.9)
    parser.add_argument('--epsilon', type=float, default=0.001)
    parser.add_argument('--epoch', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--log_period', type=int, default=1)
    parser.add_argument('--sampling', type=str, default='bernoulli')

    config = parser.parse_args()
    print(config)
    main(config)
