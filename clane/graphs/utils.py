import os

from torch_geometric.datasets import Planetoid

from clane.settings import DATA_PATH


def get_graph(dataset):
    if dataset == 'cora':
        return Planetoid(
            root=os.path.join(DATA_PATH, 'cora'),
            name='cora',
        )[0]
    else:
        raise ValueError
