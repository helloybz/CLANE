import os
import pickle

import torch

from settings import DATA_PATH


class DatasetManager:
    def get(self, dataset, device):
        X = torch.load(os.path.join(DATA_PATH, dataset, 'X.pyt'), map_location=device)
        A = pickle.load(open(os.path.join(DATA_PATH, dataset, 'A.pickle'),'rb'))
        Y = torch.load(os.path.join(DATA_PATH, dataset, 'Y.pyt'), map_location=device)
        return X, A, Y
