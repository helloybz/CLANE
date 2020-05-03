import os
import pickle

from torch import from_numpy

from .dataset import InMemoryGraphDataset


def get_graph_dataset(input_dir):
    with open(os.path.join(input_dir, "X.np"), 'rb') as X_io:
        X = pickle.load(X_io)
        X = from_numpy(X)

    with open(os.path.join(input_dir, "A.np"), 'rb') as A_io:
        A = pickle.load(A_io)
        A = from_numpy(A)

    with open(os.path.join(input_dir, "Y.pickle"), 'rb') as Y_io:
        Y = pickle.load(Y_io)

    graph = InMemoryGraphDataset(X, A, Y)

    return graph
