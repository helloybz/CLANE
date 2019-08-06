import io
import os

import torch

class DatasetManager:
    @classmethod
    def prepare_CORA(cls, f, device):
        content_io = io.open(os.path.join(f, 'cora.content'), 'r')

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
    def prepare_IMAGENET(cls, f, device):
        content_io = io.open(os.path.join(f, 'imagenet.nameurls', 'r'))

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

    def get(self, dataset, device):
        if dataset == 'cora':
            return self.prepare_CORA('data/cora', device)
        elif dataset == 'citeseer':
            return self.prepare_CITESEER('data/citeseer', device)
        elif dataset == 'imagenet':
            return self.prepare_IMAGENET('data/imagenet', device)
        else:
            raise ValueError
if __name__ == '__main__':
    f = 'data/cora'
    Preprocessor().convert_CORA(f)
