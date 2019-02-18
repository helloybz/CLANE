import argparse
import os
import pickle

import numpy as np
import torch
from scipy.spatial.distance import euclidean
from torch.distributions import Bernoulli
from torch.nn.functional import cosine_similarity
from torch.utils.data import DataLoader, Dataset

from dataset import CoraDataset, CiteseerDataset
from models import EdgeProbability
from settings import PICKLE_PATH

os.environ['CUDA_LAUNCHING_BLOCKING'] = '1'

def cosine_sim_method(c_x, edges, **kwargs):
    v_x = c_x.clone().to(kwargs['device'])
    is_converged = False

    while not is_converged:
        is_converged = True
        max_delta = -np.inf

        for doc_idx, v in enumerate(v_x):
            ref_doc_idxs = [edge[0] if edge[0] != doc_idx else edge[1]
                            for edge in edges if doc_idx in edge]
            if len(ref_doc_idxs) != 0:
                # similarities
                sims = (np.exp(cosine_similarity(v_x[doc_idx], v_x[ref_idx],
                                                 dim=0)).to(kwargs['device'])
                        for ref_idx in ref_doc_idxs)

                # regularized similarities
                r_sims = (sim / sum(sims) for sim in sims)

                try:
                    diff = config.alpha * sum([r_sim * v_x[ref_doc_idx]
                                               for ref_doc_idx, r_sim
                                               in zip(ref_doc_idxs, r_sims)])
                except RuntimeError:
                    continue

                delta = euclidean(v_x[doc_idx], c_x[doc_idx] + diff)

                # update max_delta
                max_delta = max(max_delta, delta)

                # update v_x
                v_x[doc_idx] = c_x[doc_idx] + diff

                print('[EMBED] method_1 // Maximum_error: {}, ({}/{})'.format(
                        np.round(max_delta, 4),
                        doc_idx,
                        c_x.shape[0]), end='\r')

        if max_delta < config.threshold_delta:
            is_converged = False

    return v_x


def edge_prob_method(dataset, **kwargs):
    edge_prob_model = EdgeProbability(dim=dataset.X.shape[1])
    edge_prob_model = edge_prob_model.cuda(kwargs['device'])

    dataset.Z = dataset.X.clone()

    while True:
        step1_converged = False # Initialize convergence state.
        previous_Z = dataset.Z.clone()

        with torch.no_grad():
            step1_counter = 1
            distance_list = torch.Tensor([]).cuda(kwargs['device'])

            while not step1_converged:
                max_distance = -np.inf

                for doc_id in range(dataset.Z.shape[0]):
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

        # check if Z is updated
        if torch.max(dataset.Z - previous_Z) < 0.01:
            print('Z is not updated.')
            break
        else:
            print('Z is updated. Go to step2.')

        # step 2, "Update W"

        with torch.no_grad():

            for epoch in range(config.epoch):
                J_A = torch.zeros(edge_prob_model.A.weight.shape) \
                           .to(kwargs['device'])
                J_B = torch.zeros(edge_prob_model.B.weight.shape) \
                           .to(kwargs['device'])

                if epoch % 10 == 0:
                    cost = torch.ones([1]).to(kwargs['device'])

                for z1, z2 in dataset.get_all_edges():
                    prob_edge = edge_prob_model(dataset[z1], dataset[z2])

                    is_selected = (1-prob_edge).bernoulli()
                    
                    if is_selected == 1:
                        A = edge_prob_model.A.weight
                        j_A = torch.mm(
                                torch.mm(dataset[z1].unsqueeze(-1),
                                         dataset[z2].unsqueeze(0)),
                                torch.t(A))
                        J_A.add_(j_A * is_selected)
                        if epoch % 10 == 0:
                            cost.mul_(prob_edge)

                for z1, z3s in dataset.get_all_non_edges():
                    prob_edges = edge_prob_model.forward_batch(
                                     dataset[z1].unsqueeze(0),
                                     dataset[z3s])
                    is_selected = prob_edges.bernoulli().squeeze(-1)
                    A = edge_prob_model.A.weight
                    z1, z3s = torch.broadcast_tensors(dataset[z1], dataset[z3s].squeeze(-2))
                    if epoch % 10 == 0:
                        cost.mul_(torch.prod(prob_edges[is_selected.byte()]))

                    is_selected, z3s = torch.broadcast_tensors(is_selected.unsqueeze(-1), z3s)
                    sampled_z3s = is_selected * z3s
                    j_A = torch.mm(
                            torch.mm(torch.t(z1), sampled_z3s),
                            torch.t(A))
                    J_A.sub_(j_A)

                edge_prob_model.A.weight.data.add_(config.lr * J_A)

                # 일정 에폭마다 샘플된 페어들에 대해서 코스트 계산
                if epoch % 10 == 0:
                    print('epoch: {} , cost: {}'.format(epoch, cost))

            print('step2 done')
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
        # v = method_1(c_x, edges, device=device)
        pass
    elif config.method == 2:
        dataset = edge_prob_method(dataset, device=device)
    else:
        pass

    pickle.dump(dataset.Z.cpu().data.numpy(), open(
            os.path.join(PICKLE_PATH, config.dataset,
                         'v_{}_method{}_epoch{}_batch{}_gamma{}_thr{}'.format(
                                 config.dataset,
                                 config.method,
                                 config.epoch,
                                 config.batch_size,
                                 config.gamma,
                                 config.threshold_distance,
                         )), 'wb'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--method', type=int, default=1)
    parser.add_argument('--gamma', type=float, default=0.9)
    parser.add_argument('--threshold_distance', type=float, default=0.001)
    parser.add_argument('--epoch', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--gpu', type=int, default=0)

    config = parser.parse_args()
    print(config)
    main(config)
