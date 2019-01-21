import argparse
import os
import pickle
import torch

import numpy as np
from scipy.spatial.distance import euclidean
from torch.nn.functional import cosine_similarity
from torch.utils.data import DataLoader, Dataset, TensorDataset

from dataset import CoraDataset
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
    v, ref, unref = zip(*data)
    vs = torch.stack(v)

    return vs, ref, unref


def get_content_vectors(dataset):
    if dataset == 'painter':
        # Load or Calculate C_X
        if not os.path.exists(os.path.join(PICKLE_PATH,
                                           config.dataset,
                                           'c_{}_{}'.format(
                                                   config.img_embedder,
                                                   config.doc2vec_size)
                                           )):

            print('not exists')
            raise FileExistsError
        else:
            c = pickle.load(open(os.path.join(PICKLE_PATH,
                                              config.dataset,
                                              'c_{}_{}'.format(
                                                      config.img_embedder,
                                                      config.doc2vec_size)),
                                 'rb'))
            return c


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
    sim_model = SimilarityMethod2(dim=dataset.c_x.shape[1]).to(kwargs['device'])
    v_x = dataset.c_x.clone().to(kwargs['device'])
    
    while True:
        step1_converged = False
        with torch.no_grad():
            while not step1_converged:
                max_delta = -np.inf
                step1_converged = True
                for v_idx, v in enumerate(v_x):
                    ref_idxs = list(dataset.get_refs_of(v_idx))
                        
                    if ref_idxs:
                        ref_vs = v_x[ref_idxs]
                        sims = sim_model(v_x[v_idx].unsqueeze_(0), [ref_vs])

                        messages = None
                        for ref_idx, sim in zip(ref_idxs, sims[0]):
                            print('sim', sim, sim.shape)
                            print('current embedding', v_x[ref_idx], v_x[ref_idx].shape)
                            message = sim * v_x[ref_idx]
                            if not messages:
                                messages = message 
                            else:
                                messages = messages + message

                        messages = messages * config.alpha

                        try:
                            current_embedding = v_x[v_idx]
                            updated_embedding = c_x[v_dix] + messages

                            print(updated_embedding - current_embedding)
                            delta = torch.sum(((v_x[doc_idx] - (
                                    c_x[doc_idx] + diff)) ** 2).cpu())
                        except ValueError:
                            raise

                        # update max_delta
                        max_delta = max(max_delta, delta)

                        # update v_x
                        v_x[doc_idx] = c_x[doc_idx] + diff

                        print(
                                '[EMBED] method_2 - step1 // Maximum_error: {}, ({}/{})'.format(
                                        np.round(max_delta, 4),
                                        doc_idx,
                                        c_x.shape[0]), end='\r')

                if max_delta > config.threshold_delta:
                    step1_converged = False

            print('step1 done')
        # check if v is updated
        if torch.all(torch.eq(prev_V, v_x)) == 1:
            break

        #
        # step 2, "Update W"
        #
        def loss_W(_vs, _refs, _unrefs):
            # _prob_edge 만들어서 고치기
            _loss = sum([torch.sum(torch.log(sample)) for sample in
                         sim_model(_vs, _refs)]) + \
                    sum([torch.sum(torch.log(1 - sample)) for sample in
                         sim_model(_vs, _unrefs)])
            return _loss

        sim_model.train()

        optimizer = torch.optim.Adam(params=sim_model.parameters())
        optimizer.zero_grad()

        v_dataset = VDataSet(v_x, edges)

        for epoch in range(config.epoch):
            v_loader = DataLoader(v_dataset,
                                  batch_size=config.batch_size,
                                  shuffle=True,
                                  collate_fn=v_collate_fn,
                                  )

            for batch_idx, (vs, refs, unrefs) in enumerate(v_loader):
                loss = loss_W(vs, refs, unrefs)
                loss.backward()
                optimizer.step()

                print('[EMBED] method_2 - step2// loss: {} epoch: {} batch: {}'.format(
                        loss, epoch, batch_idx),
                        end='\r')

    return v_x


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


    if config.method == 1: # Cosine similarity
        v = method_1(c_x, edges, device=device)
    elif config.method == 2: # UNNAMED
        method_2(dataset, device=device)
    else:
        pass

    pickle.dump(v, open(
            os.path.join(PICKLE_PATH, config.dataset, 'v_{}_{}'.format(
                    config.img_embedder,
                    config.doc2vec_size,
                    config.method,
                    config.alpha,
                    config.threshold_delta
            )), 'wb'))
    # print(len(edges))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--img_embedder', type=str, default='resnet152')
    parser.add_argument('--doc2vec_size', type=str, default='1024')
    parser.add_argument('--method', type=int, default=1)
    parser.add_argument('--alpha', type=float, default=0.9)
    parser.add_argument('--threshold_delta', type=float, default=0.001)
    parser.add_argument('--epoch', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=100)

    config = parser.parse_args()
    print(config)
    main(config)
