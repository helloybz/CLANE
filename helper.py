import pdb
import os
import pickle
import argparse

from torch.nn.functional import normalize

from settings import PICKLE_PATH, DATA_PATH


class DataTransformer:
    def transform_out(self, src_model, src_path, output_model, output_path):
        import torch
        import networkx as nx
        io = open(src_path, 'r')
        if output_model == 'cora':
            from dataset import CoraDataset
            dataset = CoraDataset()
            if 'sampled' in output_path:
                dataset.G = nx.read_gpickle(os.path.join(PICKLE_PATH, 'network', 'cora_X_sampled'))
        else:
            raise ValueError

        if src_model == 'deepwalk' or src_model == 'node2vec':
            io.readline()
            while True:
                sample = io.readline()
                if not sample: break
                node_id, *embedding = sample.split(' ')
                embedding = torch.tensor(([float(val.strip()) for val in embedding]))
                dataset.G.nodes[node_id]['z'] = embedding
        else:
            raise ValueError

        io.close()

        nx.write_gpickle(dataset.G, os.path.join(PICKLE_PATH, 'network', output_path))
        for node_id in dataset.G.nodes(): 
            dataset.G.nodes[node_id]['z'] = dataset.G.nodes[node_id]['z'].cpu()
        pickle.dump(dataset.Z.numpy(), open(os.path.join(DATA_PATH, output_path), 'wb'))
        
        

def normalize_elwise(*tensors):
    keep_dim = [tensor.shape for tensor in tensors]
    tensors = [tensor.flatten() for tensor in tensors]
    tensors = [normalize(tensor, dim=0) for tensor in tensors]
    tensors = [tensor.reshape(dim) for tensor, dim in zip(tensors, keep_dim)]
    return tuple(tensors)

def main(config):
    if config.task == 'transform':
        import glob
        transformer = DataTransformer()
        src_path = glob.glob(os.path.join(config.target_pattern))
        for path in src_path:
            transformer.transform_out('node2vec', path, 'cora', path)
    else:
        raise ValueError
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--task', type=str)
    parser.add_argument('--target_pattern', type=str)
    config = parser.parse_args()
    main(config)
