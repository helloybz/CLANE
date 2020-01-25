from . import Cora
from graphs import KarateClub


def get_graph(dataset):
    if dataset == 'cora':
        return Cora()
    elif dataset == 'karate':
        return KarateClub()
    else:
        raise ValueError
