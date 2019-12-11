from .graph import InMemoryDataset


class KarateClub(InMemoryDataset):
    def __init__(self):
        from torch_geometric.datasets import KarateClub as karate
        self.data = karate()[0]
        super(KarateClub, self).__init__()
