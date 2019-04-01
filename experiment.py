import argparse
import glob
import os
import pdb
import pickle

import networkx as nx 
from scipy.spatial.distance import cosine
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from torch import device

from dataset import CoraDataset
from settings import DATA_PATH
from settings import PICKLE_PATH

DATASET_MAP = {'cora':CoraDataset}

def node_classification(embeddings, labels, **kwargs):
    # split dataset
    train_X, test_X, train_Y, test_Y = train_test_split(
        embeddings, labels, test_size=kwargs['test_size'], random_state=0)
    
    if config.clf == 'nn':
        import torch
        from torch import tensor
        from torch.nn import CrossEntropyLoss
        from torch.nn import Linear
        from torch.nn import Module
        from torch.nn.functional import relu
        from torch.optim import SGD
        from torch.utils.data import DataLoader, Dataset

        in_feature = train_X.shape[1]
        number_of_classes = len(set(test_Y))
        device_ = device('cuda:0')

        one_hot_train_Y = torch.zeros(len(train_Y), number_of_classes).to(device_)
        for idx, label in enumerate(train_Y):
            one_hot_train_Y[idx][label] = 1
        
        class TrainSet(Dataset):
            def __init__(self):
                super(TrainSet, self).__init__()
                self.train_X = torch.tensor(train_X).to(device_)
                self.train_Y = torch.tensor(train_Y).long().to(device_)

            def __getitem__(self, idx):
                return self.train_X[idx], self.train_Y[idx]

            def __len__(self):
                return self.train_X.shape[0]

        class NeuralNet(Module):
            def __init__(self):
                super(NeuralNet, self).__init__()
                self.fc1 = Linear(in_feature, 500)
                self.fc2 = Linear(500, 100)
                self.fc3 = Linear(100, number_of_classes)
                
            def forward(self, x):
                x = relu(self.fc1(x))
                x = relu(self.fc2(x))
                x = self.fc3(x)
                return x

        clf = NeuralNet().to(device_)
        criterion = CrossEntropyLoss()
        optimizer = SGD(clf.parameters(), lr=0.001, momentum=0.9)

        for epoch in range(100):
            for x, y in DataLoader(TrainSet(), shuffle=True):
               optimizer.zero_grad()
               output = clf(x)
               loss = criterion(output, y)
               loss.backward()
               optimizer.step()

        with torch.no_grad():
            pred = torch.max(clf(torch.tensor(test_X).to(device_)), 1)[-1].cpu()

    elif config.clf == 'lr':
        clf = OneVsRestClassifier(
                LogisticRegression(random_state=0, solver='lbfgs', 
                                   multi_class='multinomial', max_iter=300))
        clf.fit(train_X, train_Y)
        pred = clf.predict(test_X)

    result = dict()
    result['embedding'] = kwargs['name']
    result['micro_f1'] = f1_score(pred, test_Y, average='micro')
    result['macro_f1'] = f1_score(pred, test_Y, average='macro')
    return result

def link_prediction(model_tag, **kwargs):
    target_network = nx.read_gpickle(os.path.join(PICKLE_PATH, 'network', model_tag))
    original_network = DATASET_MAP[model_tag.split('_')[0]](device=kwargs['device']).G
    
    removed_edges = original_network.edges() - target_network.edges()
    from random import sample
    negative_edges = sample(list(nx.non_edges(original_network)), len(removed_edges))

    test_edges = list(removed_edges) + negative_edges
    labels = [1]*len(removed_edges) + [0]*len(negative_edges)
    
    pred = list()
    for src, dst in test_edges: 
        z_src = target_network.nodes()[src]['z'].cpu()
        z_dst = target_network.nodes()[dst]['z'].cpu()
        pred.append(1-cosine(z_src, z_dst)) 

    result = dict()
    result['model_tag'] = model_tag
    result['AUC'] = roc_auc_score(labels, pred)
    return result

def main(config):
    device_ = device('cuda')
    
    # load target embeddings. 
    target_paths = glob.glob(os.path.join(DATA_PATH, 
                                          'experiments', 'target',
                                          config.experiment, '*'))

    if config.experiment == 'node_classification':
        for target_path in target_paths:
            labels = DATASET_MAP[os.path.basename(target_path).split('_')[0]](device=device_).Y.numpy()
            try:
                embedding = pickle.load(open(target_path, 'rb'))
            except UnicodeDecodeError:
                embedding = pickle.load(open(target_path, 'rb'), encoding='latin-1')

            result = node_classification(embedding, labels, test_size=config.test_size,
                                         name=os.path.basename(target_path)) 
            print('===========================')
            for key in result.keys():
                if key == 'embedding':
                    print(result[key])
                else:
                    print(key, '\t', '{:4f}'.format(result[key]))
    elif config.experiment == 'link_prediction':
        target_paths = [target_path for target_path in target_paths 
                        if 'sampled' in target_path]
        for target_path in target_paths:
            model_tag = os.path.basename(target_path)
            result = link_prediction(model_tag=model_tag, device=device_)
       
            print('===========================')
            for key in result.keys():
                if key == 'model_tag':
                    print(result[key])
                else:
                    print(key, '\t', '{:4f}'.format(result[key]))

    else:
        raise ValueError


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--experiment', type=str, default='node_classification')
    parser.add_argument('--dataset', type=str, default='cora')
    parser.add_argument('--test_size', type=float, default='0.4')
   
    parser.add_argument('--clf', type=str, default='nn')
    config = parser.parse_args()
    print(config)
    main(config)

