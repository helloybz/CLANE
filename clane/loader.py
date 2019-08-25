import io
import os
import pickle

import torch

from settings import DATA_PATH


class DatasetManager:
    @classmethod
    def prepare_CORA(cls, dataset, directed, device):
        content_io = io.open(os.path.join(DATA_PATH, dataset, 'cora.content'), 'r')

        node_ids = []
        contents = []
        Y = []

        while True:
            line = content_io.readline()
            if line == '': break
            node_id, *content, label = line.split()
            node_ids.append(node_id)
            contents.append([float(val) for val in content])
            Y.append(label)
        
        n, d = len(contents), len(contents[0])
        X = torch.tensor(contents, device=device)

        A = {i:[] for i in range(len(node_ids))}
        edge_io = io.open(os.path.join(DATA_PATH, dataset, 'cora.cites'), 'r')
        while True:
            cite = edge_io.readline()
            if cite == '': break
            dst, src = cite.split()
            dst, src = node_ids.index(dst), node_ids.index(src)
            A[src].append(dst)
            if not directed:
                A[dst].append(src)

        label_set = list(set(Y))
        Y = torch.tensor([label_set.index(label) for label in Y], device=device)
        
        return node_ids, X, A, Y

    @classmethod
    def prepare_IMAGENET(cls, f, device):
        content_io = io.open(os.path.join(f, 'imagenet.nameurls'), 'r')

        image_names = []
        image_urls = []
        node_ids = []
        contents = []
        Y = []

        while True:
            line = content_io.readline()
            if line == '': break
            node_id, *content, label = line.split()
            node_ids.append(node_id)
            contents.append([float(val) for val in content])
            Y.append(label)
        
        n, d = len(contents), len(contents[0])
        A = torch.zeros(n, n)
        X = torch.tensor(contents)

        edge_io = io.open(os.path.join(f, 'cora.cites'), 'r')
        while True:
            cite = edge_io.readline()
            if cite == '': break
            dst, src = cite.split()
            dst, src = node_ids.index(dst), node_ids.index(src)
            A[src, dst] = 1

        label_set = list(set(Y))
        Y = torch.tensor([label_set.index(label) for label in Y])
        
        return node_ids, X.to(device), A.to(device), Y.to(device)
        
    @classmethod
    def prepare_PPI(self, dataset, directed, device):
        import numpy as np 
        X = np.load(os.path.join(DATA_PATH, dataset, 'ppi-feats.npy'))
        X = torch.tensor(X, device=device, dtype=torch.float)

        node_ids = [str(number) for number in range(X.shape[0])]
        A = {i:[] for i in range(len(node_ids))}

        import json
        G = json.load(open(os.path.join(DATA_PATH, dataset, 'ppi-G.json')))
        for link in G['links']:
            source, target = link['source'], link['target']
            if source == target: continue
            A[source].append(target)
            if not directed:
                A[target].append(source)

        Y = torch.zeros(len(node_ids),121, device=device)
        label_dict = json.load(open(os.path.join(DATA_PATH, dataset, 'ppi-class_map.json'),'r'))
        for key in label_dict.keys():
            idx = node_ids.index(key)
            Y[idx] = torch.tensor(label_dict[key], device=device)
        return node_ids, X, A, Y

    def get(self, dataset):
        X = torch.load(os.path.join(DATA_PATH, dataset, 'X.pyt'))
        A = pickle.load(open(os.path.join(DATA_PATH, dataset, 'A.pickle'),'rb'))
        Y = torch.load(os.path.join(DATA_PATH, dataset, 'Y.pyt'))
        return X, A, Y
#        if dataset == 'cora':
#            return self.prepare_CORA(dataset, directed, device)
#        elif dataset == 'citeseer':
#            return self.prepare_CITESEER(dataset, device)
#        elif dataset == 'imagenet':
#            return self.prepare_IMAGENET(dataset, directed, device)
#        elif dataset == 'ppi':
#            return self.prepare_PPI(dataset, directed, device)
#        else:
#            raise ValueError

if __name__ == '__main__':
    f = 'data/cora'
    Preprocessor().convert_CORA(f)
