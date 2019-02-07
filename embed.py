import argparse
import os
import pickle

import numpy as np
import torch
from scipy.spatial.distance import euclidean
from torch.nn.functional import cosine_similarity
from torch.utils.data import DataLoader, Dataset

from dataset import CoraDataset
from dataset import cora_collate
from models import SimilarityMethod2
from settings import PICKLE_PATH


def method_1(c_x, edges, **kwargs):
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


def method_2(dataset, **kwargs):
    similarity_model = SimilarityMethod2(dim=dataset.X.shape[1])
    similarity_model = similarity_model.cuda(kwargs['device'])

    dataset.Z = dataset.X.clone()

    while True:
        step1_converged = False
        previous_Z = dataset.Z.clone()

        with torch.no_grad():
            step1_counter = 1
            while not step1_converged:
                max_distance = -np.inf
                step1_converged = True

                for doc_id, _ in enumerate(dataset.Z):
                    Z_ref = dataset.Z[(dataset.A[doc_id, :] + dataset.A[:, doc_id]).byte()]
                    similarities = similarity_model(dataset.Z[doc_id], Z_ref)
                    message = config.gamma * torch.mv(torch.t(Z_ref), similarities)

                    updated_z = dataset.X[doc_id] + message

                    distance = dataset.Z[doc_id] - updated_z
                    distance = torch.sum(distance ** 2)

                    max_distance = max(max_distance, distance)

                    dataset.Z[doc_id] = updated_z

                    print(
                        'Method2 / UpdateZ / {0} / maximum_loss {1} / {2}/{3}'.format(
                            step1_counter, max_distance, doc_id, dataset.Z.shape[0]),
                        end='\r')

                if max_distance > config.threshold_distance:
                    step1_converged = False
                step1_counter += 1
            print('step1 done')

        # check if Z is updated
        if torch.all(torch.eq(previous_Z, Z)) == 1:
            print('Z is not updated.')
            break

        dataset.Z = dataset.X.clone()

        # step 2, "Update W"
        def loss_W(_zs, _refs_batch, _unrefs_batch):
            batch_loss = None
            for idx, _z in enumerate(_zs):
                if len(_refs_batch[idx]) != 0:
                    ref_probs_edge = [torch.log(similarity_model.prob_edge(_z, _ref))
                                      for _ref
                                      in _refs_batch[idx]]
                    ref_loss = torch.sum(torch.stack(ref_probs_edge), dim=0)
                else:
                    ref_loss = 0

                if len(_unrefs_batch[idx]) != 0:
                    unref_probs_edge = [torch.log(1 - similarity_model.prob_edge(_z, _unref))
                                        for _unref
                                        in _unrefs_batch[idx]]
                    unref_loss = torch.sum(torch.stack(unref_probs_edge), dim=0)
                else:
                    unref_loss = 0

                # print(ref_loss.shape, unref_loss.shape)
                if batch_loss is not None:
                    batch_loss += (ref_loss + unref_loss)
                else:
                    batch_loss = (ref_loss + unref_loss)

            _loss = -(batch_loss / config.batch_size)
            return _loss

        similarity_model.train()

        optimizer = torch.optim.Adam(params=similarity_model.parameters())
        optimizer.zero_grad()

        for epoch in range(config.epoch):
            dataloader = DataLoader(dataset,
                                    batch_size=config.batch_size,
                                    shuffle=True,
                                    collate_fn=cora_collate,
                                    )
            optimizer.zero_grad()
            for batch_idx, (zs, ref, unref) in enumerate(dataloader):
                loss = loss_W(zs, ref, unref)
                loss.backward()
                optimizer.step()

                print('Method2 / UpdateA / epoch: {} / batch: {} / loss: {}'.format(
                        epoch, batch_idx, loss.data[0]), end='\r')

    return Z


def main(config):
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print('[DEVICE] {}'.format(device))

    if config.dataset == 'cora':
        dataset = CoraDataset(device=device)
    else:
        raise ValueError
    print('DATASET LOADED.')

    if config.method == 1:
        # v = method_1(c_x, edges, device=device)
        pass
    elif config.method == 2:
        Z = method_2(dataset, device=device)
    else:
        pass

    pickle.dump(Z, open(
            os.path.join(PICKLE_PATH, config.dataset, 'v_{}'.format(
                    config.dataset,
            )), 'wb'))

    pickle.dump(Z, open(
            os.path.join(PICKLE_PATH, config.dataset, 'v_{}_{}_{}_{}_{}_{}'.format(
                    config.dataset,
                    config.method,
                    config.epoch,
                    config.batch_size,
                    config.gamma,
                    config.threshold_distance,
            )), 'wb'))
    # print(len(edges))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--directed', type=bool, default=False)
    parser.add_argument('--method', type=int, default=1)
    parser.add_argument('--gamma', type=float, default=0.9)
    parser.add_argument('--threshold_distance', type=float, default=0.001)
    parser.add_argument('--epoch', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=100)

    config = parser.parse_args()
    print(config)
    main(config)
