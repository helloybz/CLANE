import os
from random import sample

from torch.utils.data import Dataset, DataLoader
import torch

from settings import DATA_PATH


class CoraDataset(Dataset):
    def __init__(self):
        self.paper_ids = list()
        self.labels = list()
        self.X = None
        self.A = None
        self.Z = None

        with open(os.path.join(DATA_PATH, 'cora', 'cora.content'),
                  'r') as cora_content_io:
            while True:
                sample = cora_content_io.readline()

                if not sample:
                    break

                paper_id, *content, label = sample.split('\t')
                self.paper_ids.append(int(paper_id))
                self.labels.append(label)
                content = torch.Tensor([[int(value) for value in content]])

                if self.X is not None:
                    self.X = torch.cat([self.X, content], 0)
                else:
                    self.X = content

        self.A = torch.zeros(len(self.paper_ids), len(self.paper_ids))

        with open(os.path.join(DATA_PATH, 'cora', 'cora.cites'),
                  'r') as cora_edge_io:
            while True:
                sample = cora_edge_io.readline()
                if not sample:
                    break

                cited, citing = sample.split('\t')
                cited = self.paper_ids.index(int(cited))
                citing = self.paper_ids.index(int(citing))
                self.A[cited, citing] = 1

    def __getitem__(self, index):
        ref_idx = [idx 
                   for idx, elem 
                   in enumerate(list(self.get_ref_ids(index)))
                   if elem == 1]
        sampled_ref_Z = self.Z[ref_idx]

        unref_idx = [idx
                     for idx, elem
                     in enumerate(list(self.get_ref_ids(index)))
                     if elem == 0]
        if ref_idx:
            unref_idx = sample(unref_idx, len(ref_idx))
        else:
            unref_idx = sample(unref_idx, 10)
        sampled_unref_Z = self.Z[unref_idx]

        return self.Z[index], sampled_ref_Z, sampled_unref_Z

    def __len__(self):
        return len(self.paper_ids)

    def get_ref_ids(self, doc_idx, directed=False):
        outgoing_refs = self.A[:, doc_idx]
        ingoing_refs = self.A[doc_idx, :]

        if directed:
            return ingoing_refs, outgoing_refs
        else:
            return outgoing_refs + ingoing_refs


def cora_collate(data):
    z, ref, unref = zip(*data)
    zs = torch.stack(z)

    return zs, ref, unref


class CiteseerDataset(Dataset):
    def __init__(self):
        self.paper_ids = list()
        self.labels = list()
        self.X = None
        self.A = None
        self.Z = None

        with open(os.path.join(DATA_PATH, 'citeseer', 'citeseer.content'),
                  'r') as citeseer_content_io:
            while True:
                sample = citeseer_content_io.readline()

                if not sample:
                    break

                paper_id, *content, label = sample.split('\t')
                self.paper_ids.append(paper_id)
                self.labels.append(label)
                content = torch.Tensor([[int(value) for value in content]])

                if self.X is not None:
                    self.X = torch.cat([self.X, content], 0)
                else:
                    self.X = content

        self.A = torch.zeros(len(self.paper_ids), len(self.paper_ids))

        with open(os.path.join(DATA_PATH, 'citeseer', 'citeseer.cites'),
                  'r') as citeseer_edge_io:
            while True:
                sample = citeseer_edge_io.readline()
                if not sample:
                    break

                cited, citing = sample.split('\t')
                cited = self.paper_ids.index(cited)
                citing = self.paper_ids.index(citing)
                self.A[cited, citing] = 1

    def __getitem__(self, index):
        ref_idx = [idx
                   for idx, elem
                   in enumerate(list(self.get_ref_ids(index)))
                   if elem == 1]
        sampled_ref_Z = self.Z[ref_idx]

        unref_idx = [idx
                     for idx, elem
                     in enumerate(list(self.get_ref_ids(index)))
                     if elem == 0]
        if ref_idx:
            unref_idx = sample(unref_idx, len(ref_idx))
        else:
            unref_idx = sample(unref_idx, 10)
        sampled_unref_Z = self.Z[unref_idx]

        return self.Z[index], sampled_ref_Z, sampled_unref_Z

    def __len__(self):
        return len(self.paper_ids)

    def get_ref_ids(self, doc_idx, directed=False):
        outgoing_refs = self.A[:, doc_idx]
        ingoing_refs = self.A[doc_idx, :]

        if directed:
            return ingoing_refs, outgoing_refs
        else:
            return outgoing_refs + ingoing_refs



if __name__ == "__main__":
    cora = CoraDataset()
    print(cora[3])
    print(len(cora))
