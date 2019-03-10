import argparse
import os
import pdb
import pickle
import time

import numpy as np
from tensorboardX import SummaryWriter
import torch
from torch.distributions import Bernoulli
from torch.nn import CosineSimilarity
from torch.utils.data import DataLoader, Dataset

from dataset import CoraDataset, CiteseerDataset
from helper import  normalize_elwise
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
    writer = SummaryWriter(log_dir='runs/{}'.format(config.model_tag))
    # torch.set_grad_enabled(False)

    if config.model_load is not None:
        edge_prob_model = torch.load(os.path.join(PICKLE_PATH, 'models', 
                    config.model_load))
        checkpoint = torch.load(os.path.join(PICKLE_PATH, 'checkpoints', config.model_load))
        iter_counter = checkpoint['iter_counter']
        dataset.Z = checkpoint['dataset_Z']

    else:
        edge_prob_model = EdgeProbability(dim=dataset.X.shape[1])
        iter_counter = 0
        dataset.Z = dataset.X.clone()

    edge_prob_model = edge_prob_model.cuda(kwargs['device'])

    for p in edge_prob_model.parameters(): p.requires_grad = False
    
    while True:
        iter_counter += 1

        step1_converged = False
        step1_counter = 1
        delta_list = []

        step1_start = time.time()
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
                newly_calculated_z = dataset.X[doc_id] + config.gamma * messages

                delta = torch.norm(newly_calculated_z - z, 2)
                maximum_delta = max(maximum_delta, delta)

                dataset.Z[doc_id] = newly_calculated_z

                print('UpdateZ / {0} / maximum_distance {1:.6f} / {2:4d}/{3:4d}'\
                      .format(step1_counter, 
                              maximum_delta, 
                              doc_id,
                              len(dataset)),
                      end='\r')
                
            # UpdateZ done
            
            delta_list.append(maximum_delta)

            if len(delta_list) == 10:
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
        writer.add_scalar('{}/Convergence time Updating Z'.format(config.model_tag),
                          step1_time[0], iter_counter)
        print("Z convergence time: {:.3f}".format(step1_end - step1_start))

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
                                          'temp_v_{}_method{}_epoch{}_batch{}_gamma{}_thr{}_{}'.format(
                                             config.dataset,
                                             config.method,
                                             config.epoch,
                                             config.batch_size,
                                             config.gamma,
                                             config.epsilon,
                                             delta_Z)), 
                             'wb'))
            
        # step 2, "Update W"
        step2_start = time.time()
        for epoch in range(config.epoch):
            step2_epoch_start = time.time()
            
            J_A = torch.zeros(edge_prob_model.A.weight.shape) \
                       .to(kwargs['device'])
            J_B = torch.zeros(edge_prob_model.B.weight.shape) \
                       .to(kwargs['device'])
            
            if epoch % config.log_period == 0:
                cost = torch.zeros([1]).to(kwargs['device'])
             
#            for doc_id, ref_id in dataset.get_all_edges():
#                prob_edge = edge_prob_model(dataset[doc_id], dataset[ref_id])
#                
#                is_selected = prob_edge.bernoulli()
#                
#                if is_selected == 1:
#                    
#                    j_A = torch.mm(dataset[ref_id].unsqueeze(-1),
#                                   dataset[doc_id].unsqueeze(0))
#                    j_A = torch.mm(edge_prob_model.B.weight.clone(),
#                                   j_A)
#                    
#                    j_B = torch.mm(dataset[ref_id].unsqueeze(-1),
#                                   dataset[doc_id].unsqueeze(0))
#                    j_B = torch.mm(j_B,
#                                   torch.t(edge_prob_model.A.weight.clone()))
#                    
#                    J_A.add_(j_A)
#                    J_B.add_(j_B)
#
#                    if epoch % config.log_period == 0:
#                        cost.sub_(torch.log(prob_edge))                    
                
                print("Update Params edge pair: {:4d}/{:4d}".format(doc_id, len(dataset)), end='\r') 
            
            for doc_id, unref_ids in dataset.get_all_non_edges():
                prob_edges = edge_prob_model.forward_batch(
                                 dataset[doc_id].unsqueeze(0),
                                 dataset[unref_ids]).squeeze()
                
                if config.sampling == 'bernoulli':
                    is_selected = prob_edges.bernoulli().squeeze()
                elif config.sampling.startswith('top'):
                    number_of_sample = int(config.sampling.split('top')[-1])
                    is_selected = prob_edges.sort(dim=0)[1][:number_of_sample].squeeze()
                else:
                    raise ValueError
                
                z = dataset[doc_id]
                Z_unref = dataset[unref_ids.squeeze()][is_selected]
        
                for indices in DataLoader(range(Z_unref.shape[0]),
                                          batch_size=400):
                    j_A = torch.matmul(z.unsqueeze(-1),
                                       Z_unref[indices].unsqueeze(-2))

                    j_A = torch.matmul(edge_prob_model.B.weight,
                                       j_A)
                    j_A = torch.sum(j_A, dim=0)
                    J_A.sub_(j_A)

                    j_B = torch.matmul(Z_unref[indices].unsqueeze(-1),
                                       z.unsqueeze(0))
                    j_B = torch.transpose(j_B, 1, 2)
                    j_B = edge_prob_model.A(j_B)
                    j_B = torch.transpose(j_B, 1, 2)
                    j_B = torch.sum(j_B, dim=0)
                    J_B.sub_(j_B)
               
                if epoch % config.log_period == 0:
                    cost.sub_(torch.sum(torch.log(1-prob_edges[is_selected])))

                print("Update Params non-edge pair: {:4d}/{:4d}".format(doc_id, len(dataset)), end='\r')

            
            # normalize gradients.
            J_A, J_B = normalize_elwise(J_A, J_B)
            edge_prob_model.A.weight.data.add_(config.lr * J_A)
            edge_prob_model.B.weight.data.add_(config.lr * J_B)

            # 일정 에폭마다 샘플된 페어들에 대해서 코스트 계산
            if epoch % config.log_period == 0:
                print('epoch: {} , cost: {}'.format(epoch, cost))
                writer.add_scalar('{}/Cost'.format(config.model_tag),
                                  cost[0], ((iter_counter-1) * config.epoch) + epoch)
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
    import datetime
    model_tag = str(datetime.datetime.today().isoformat('-')).split('.')[0]
    parser.add_argument('--model_tag', type=str, default=model_tag)
    parser.add_argument('--model_load', type=str)

    config = parser.parse_args()
    print(config)
    main(config)
