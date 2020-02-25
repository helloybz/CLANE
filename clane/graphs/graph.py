import os

import torch
from torch.utils.data import Dataset

from manager import ContextManager


class GraphDataset(Dataset):
    def __init__(self):
        super(GraphDataset, self).__init__()
        self.node_traversal = False
        '''
        node_traversal (Boolean):
            If True, the __getitem__ method will return
            all the stuffs which are needed to optimize
            the node embeddings.
            Otherwise, the method will return a pair of
            nodes which is needed to training the tran-
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
            return self.get_z([i, j]), edge.float()

    @torch.no_grad()
    def build_transition_matrix(self, similarity):
        for idx, indices in enumerate(self.edge_index.t()):
            z = self.get_z(indices).to(
                ContextManager.instance().device, non_blocking=True)
            self.edge_similarity[idx] = similarity(
                *z.split(split_size=1, dim=0))

    def get_x(self, index):
        raise NotImplementedError

    def get_z(self, index):
        raise NotImplementedError

    def set_z(self, index, values):
        raise NotImplementedError


class InMemoryDataset(GraphDataset):
    def __init__(self):
        super(InMemoryDataset, self).__init__()
        self.x = self.data.x
        self.y = self.data.y
        self.edge_index = self.data.edge_index
        self.edge_similarity = self.x.new_ones(self.edge_index[0].shape)
        self.z = self.x.clone()
        del self.data

    def get_x(self, index):
        return self.x[index]

    def get_z(self, index):
        return self.z[index]

    def set_z(self, index, values):
        self.z[index] = values.cpu()

    @property
    def num_nodes(self):
        return self.x.shape[0]

    @property
    def feature_dim(self):
        return self.x.shape[1]


class OnDiskDataset(GraphDataset):
    def __init__(self):
        super(OnDiskDataset, self).__init__()
        # TODO: Prepare x paths
        # TODO: Prepare z paths
        # TODO: Prepare edge paths or an edge_index tensor
        # TODO: Read meta data from the file
        self.feature_dim, self.num_nodes = 0, 0

    def get_x(self, index):
        with open(os.path.join(self.x_path, f'{index}.csv')) as f:
            x = torch.tensor(f.read().split(','), dtype=torch.float64)
        return x

    def get_z(self, index):
        with open(os.path.join(self.z_path, f'{index}.csv')) as f:
            z = torch.tensor(f.read().split(','), dtype=torch.float64)
        return z

    def set_z(self, index, values):
        with open(os.path.join(self.z_path, f'{index}.csv'), 'w') as f:
            f.write(','.join([str(x) for x in values.tolist()]))

    @property
    def num_nodes(self):
        return self.num_nodes

    @property
    def feature_dim(self):
        return self.feature_dim
