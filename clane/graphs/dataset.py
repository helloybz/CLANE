import os

import torch
from torch.utils.data import Dataset
from torchvision import transforms

from clane import g
from .transforms import Standardazation


class GraphDataset(Dataset):
    def __init__(self):
        super(GraphDataset, self).__init__()
        self.node_traversal = False
        '''
            If node_traversal is True, the __getitem__ 
            method will return all the stuffs which are
            needed for optimizing the node embeddings.
            Otherwise, the method will return a pair of
            nodes which is needed for training the tran-
            sition matrix.
        '''

    @property
    def num_nodes(self):
        raise NotImplementedError

    @property
    def num_edges(self):
        return self.edge_index.shape[1]

    @property
    def feature_dim(self):
        return NotImplementedError

    def __len__(self):
        if self.node_traversal:
            return self.num_nodes
        else:
            return self.num_edges

    def __getitem__(self, index):
        if self.node_traversal:
            # Used when iterating over the nodes
            # Returns all the stuffs needed for optimizing Z.
            edge_indices = self.edge_index[0] == index
            nbr_nodes = self.edge_index[1, self.edge_index[0] == index]
            return (self.get_x(index),
                    self.get_z(index),
                    self.edge_similarity[edge_indices],
                    self.get_z(nbr_nodes))
        else:
            # Iterate over the edges
            # and return all the stuffs needed for training ∏.
            # Compute src and dst from index.
            i, j = index // self.num_nodes, index % self.num_nodes

            edge = (self.edge_index[0] == i)\
                .mul(self.edge_index[1] == j)\
                .any()

            # TODO: In case of large-scale graphs,
            # prepare a method which loads the embeddings
            # by reading from the disk.
            # And call the method here instead of indexing z.
            return self.transform(self.get_z([i, j])), edge.float()

    @torch.no_grad()
    def build_transition_matrix(self, similarity):
        '''
            It actually computes the similarities between positive node pairs.
        '''
        for idx, indices in enumerate(self.edge_index.t()):
            z = self.transform(self.get_z(indices)).to(
                g.device, non_blocking=True)
            self.edge_similarity[idx] = similarity(
                *z.split(split_size=1, dim=0))

    def get_x(self, index):
        raise NotImplementedError

    def get_z(self, index):
        raise NotImplementedError

    def set_z(self, index, values):
        raise NotImplementedError


class InMemoryGraphDataset(GraphDataset):
    def __init__(self, X, A, Y):
        super(InMemoryGraphDataset, self).__init__()
        self.X = X
        self.A = A
        self.Y = Y
        self.node_similarity = self.A
        self.edge_similarity = self.X.new_ones(self.A[0].shape)
        self.Z = self.X.clone()

    def get_x(self, index):
        return self.X[index]

    def get_z(self, index):
        return self.z[index]

    def set_z(self, index, values):
        self.z[index] = values.cpu()

    def make_standard(self):
        self.transform = transforms.Compose([
            Standardazation(mean=self.z.mean(), std=self.z.std())
        ])

    @property
    def num_nodes(self):
        return self.X.shape[0]

    @property
    def feature_dim(self):
        return self.X.shape[1]
