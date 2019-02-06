import argparse
import os
import pickle
import torch

import numpy as np
from scipy.spatial.distance import euclidean
from torch.nn.functional import cosine_similarity
from torch.utils.data import DataLoader, Dataset, TensorDataset

from dataset import CoraDataset
from dataset import cora_collate
from models import SimilarityMethod2
from settings import PICKLE_PATH


class VDataSet(Dataset):

    def __init__(self, v_tensor, edges):
        # super(VDataSet, self).__init__(self)
        self.v = v_tensor
        self.edges = edges

    def __getitem__(self, index):
        refs, unrefs = self._get_refs_and_unrefs(index)
        return self.v[index], refs, unrefs

    def _get_refs_and_unrefs(self, doc_idx):
        ref_idxs = [edge[0] if edge[0] != doc_idx else edge[1]
                    for edge in self.edges if doc_idx in edge]

        unref_idxs = np.random.choice(
                [idx for idx in range(self.v.shape[0]) if idx not in ref_idxs],
                len(ref_idxs), replace=False)
        return self.v[ref_idxs], self.v[unref_idxs]

    def __len__(self):
        return self.v.size(0)


def v_collate_fn(data):
    z, ref, unref = zip(*data)
    vs = torch.stack(v)

    return vs, ref, unref


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
    similarity_model = SimilarityMethod2(dim=dataset.X.shape[1]).to(
            kwargs['device'])

    Z = dataset.X.clone().to(kwargs['device'])

    while True:
        # step1_converged = False
        # previous_Z = Z.clone()
        # with torch.no_grad():
        #     step1_counter = 0
        #     while not step1_converged:
        #         max_distance = -np.inf
        #         step1_converged = True
        #
        #         for doc_id, z in enumerate(Z):
        #             ref_ids = dataset.get_ref_ids(doc_id, directed=False)
        #
        #             # aggregate
        #             ref_ids = [ref_idx
        #                        for ref_idx, flag
        #                        in enumerate(ref_ids)
        #                        if flag == 1]
        #
        #             similarities = similarity_model(z, Z[ref_ids])
        #
        #             message = config.alpha * sum([z * sim
        #                                           for z, sim
        #                                           in zip(Z[ref_ids],
        #                                                  similarities)])
        #             # print(message)
        #
        #             # Check convergence
        #             try:
        #                 distance = Z[doc_id].to(kwargs['device']) \
        #                            - dataset.X[doc_id].to(kwargs['device']) \
        #                            - message.to(kwargs['device'])
        #             except AttributeError:
        #                 if not ref_ids:
        #                     distance = Z[doc_id].to(kwargs['device']) \
        #                                - dataset.X[doc_id].to(kwargs['device'])
        #                 else:
        #                     raise
        #
        #             distance = sum(distance ** 2)
        #             # print("distance: ", distance)
        #             max_distance = max(max_distance, distance)
        #             # print("max distance:", max_distance)
        #
        #             # Update z
        #             Z[doc_id] = dataset.X[doc_id].to(kwargs['device']) + message
        #
        #             print('Method2 / UpdateZ / {0} / maximum_loss {1} / {2}/{3}'.format(step1_counter, max_distance, doc_id, Z.shape[0]), end='\r')
        #
        #         step1_counter += 1
        #
        #         if max_distance > config.threshold_distance:
        #             step1_converged = False
        #     dataset.Z = Z
        #     print('step1 done')
        #
        # # check if Z is updated
        # if torch.all(torch.eq(previous_Z, Z)) == 1:
        #     print('Z is not updated.')
        #     break

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
        dataset = CoraDataset()
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
            os.path.join(PICKLE_PATH, config.dataset, 'v_{}_{}'.format(
                    config.dataset,
                    config.method,
                    config.epoch,
                    config.batch_size,
                    config.alpha,
                    config.threshold_distance,
            )), 'wb'))
    # print(len(edges))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--img_embedder', type=str, default='resnet152')
    parser.add_argument('--doc2vec_size', type=str, default='1024')
    parser.add_argument('--method', type=int, default=1)
    parser.add_argument('--alpha', type=float, default=0.9)
    parser.add_argument('--threshold_distance', type=float, default=0.001)
    parser.add_argument('--epoch', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=100)

    config = parser.parse_args()
    print(config)
    main(config)
