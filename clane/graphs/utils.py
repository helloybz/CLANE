from . import Cora
from graphs import KarateClub


def get_graph(dataset):
    if dataset == 'CORA':
        return Cora()
    elif dataset == 'KARATE':
        return KarateClub()
    else:
        raise ValueError
