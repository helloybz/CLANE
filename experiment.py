import argparse
import os
import pdb
import pickle

from sklearn import svm
import torch

from dataset import CoraDataset
from settings import DATA_PATH
from settings import PICKLE_PATH


def node_classification(embeddings, labels, **kwargs):
    classifier = svm.SVC(gamma='scale')
    pdb.set_trace()
    return result

def main(config):
    device = torch.device('cpu')
    # load target embeddings. 
    
    if config.vanilla:
        if config.dataset == 'cora':
            dataset = CoraDataset(device=device)
            embeddings, labels = dataset.X, dataset.labels
        else:
            raise ValueError

    else:
        # TODO: load embeddings
        pass
    
    # do experiment.
    if config.experiment == 'node_classification':
        result = node_classification(embeddings, labels)
    else:
        raise ValueError

    # summarize and visualize results.
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--experiment', type=str, default='node_classification')
    parser.add_argument('--dataset', type=str, default='node_classification')
    parser.add_argument('--vanilla', action='store_true')
    
    config = parser.parse_args()
    print(config)
    main(config)

