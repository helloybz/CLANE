import os
from random import sample, shuffle

import torch
from torch.nn.functional import pad
from torch.utils.data import Dataset

from settings import DATA_PATH


class CoraDataset(Dataset):
    def __init__(self, **kwargs):
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

        self.X = self.X.to(kwargs['device'])
        self.A = self.A.to(kwargs['device'])
        self.Z = self.X.clone()

    def __getitem__(self, index):
        return self.Z[index]

    def __len__(self):
        return len(self.paper_ids)

    def get_all_edges(self):
        for pair in self.A.nonzero():
            yield pair

    def get_all_non_edges(self):
        for z1, row in enumerate(self.A):
            yield z1, (row == 0).nonzero()


def cora_collate(data):
    batch_z, batch_z_ref, batch_z_unref = zip(*data)

    batch_z = torch.stack(batch_z)

    max_row_ref = max([ref_z.shape[0] for ref_z in batch_z_ref])
    max_row_unref = max([unref_z.shape[0] for unref_z in batch_z_unref])

    mask_ref = torch.Tensor([z_ref.shape[0] for z_ref in batch_z_ref]).cuda()
    mask_unref = torch.Tensor([unref_z.shape[0] for unref_z in batch_z_unref]).cuda()

    batch_ref_z = torch.stack(
            [pad(ref_z, [0, 0, 0, max_row_ref - ref_z.shape[0]])
             for ref_z in batch_z_ref])

    batch_unref_z = torch.stack(
            [pad(ref_z, [0, 0, 0, max_row_unref - ref_z.shape[0]])
             for ref_z in batch_z_unref])

    return batch_z, batch_ref_z, batch_unref_z


class CiteseerDataset(Dataset):
    def __init__(self, **kwargs):
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

                cited, citing = sample.strip().split('\t')
                try:
                    cited = self.paper_ids.index(cited)
                    citing = self.paper_ids.index(citing)
                    self.A[cited, citing] = 1
                except ValueError:
                    pass
        self.X = self.X.to(kwargs['device'])
        self.A = self.A.to(kwargs['device'])

    def __getitem__(self, index):
        ref_idx = (self.A[:, index] + self.A[index, :]).byte()
        unref_idx = ref_idx == 0
        shuffle(unref_idx)
        unref_idx = unref_idx[:10]
        ref_Z = self.Z[ref_idx]
        unref_Z = self.Z[unref_idx]

        return self.Z[index], ref_Z, unref_Z

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
    dataset = CiteseerDataset()
