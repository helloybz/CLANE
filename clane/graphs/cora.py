from .graph import InMemoryDataset

from clane.settings import DATA_PATH


class Cora(InMemoryDataset):
    def __init__(self):
        import os
        from torch_geometric.datasets import Planetoid
        self.data = Planetoid(
            root=os.path.join(DATA_PATH, 'cora'),
            name='Cora'
        )[0]
        super(Cora, self).__init__()
