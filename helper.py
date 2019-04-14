import pdb
import os
import pickle
from multiprocessing.pool import ThreadPool

import requests
from PIL import Image
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3 import Retry
import numpy as np
from torch.nn.functional import normalize

from settings import PICKLE_PATH, DATA_PATH


class DataTransformer:
    def transform(self, networkx, target_model):
        io = open('{}.cites'.format(target_model), 'w') 
        for src, nbrs in networkx.G.adjacency():
            line = ''.join([i+' ' for i in [src]+list(nbrs.keys())])
            io.write(line + '\n')
        io.close() 

    def transform_out(self, dataset, model, src_path, output_path):
        import torch
        import pdb; pdb.set_trace()
        io = open(src_path, 'r')
        io.readline()
        if model == 'deepwalk':
            while True:
                sample = io.readline()
                if not sample: break
                node_id, *embedding = sample.split(' ')
                embedding = torch.tensor(([float(val.strip()) for val in embedding]))
                dataset.G.nodes[node_id]['z'] = embedding
        io.close()
        
        pickle.dump(dataset.Z.numpy(), open(os.path.join(DATA_PATH, output_path), 'wb'))
        
        

def normalize_elwise(*tensors):
    keep_dim = [tensor.shape for tensor in tensors]
    tensors = [tensor.flatten() for tensor in tensors]
    tensors = [normalize(tensor, dim=0) for tensor in tensors]
    tensors = [tensor.reshape(dim) for tensor, dim in zip(tensors, keep_dim)]
    return tuple(tensors)


if __name__ == '__main__':
    transformer = DataTransformer()
    import torch
    from dataset import CoraDataset
    cora_network = CoraDataset(device=torch.device('cpu'))
    src_path = os.path.join(DATA_PATH, 'experiments', 'target' ,'node_classification', 'cora_deepwalk_d1433')
    transformer.transform_out(cora_network, 'deepwalk', src_path, 'test')
