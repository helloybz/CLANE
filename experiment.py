import argparse
import glob
import os
import pickle

import networkx as nx 
from scipy.spatial.distance import cosine
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from torch import device
import torch

from dataset import CoraDataset
from settings import DATA_PATH
from settings import PICKLE_PATH

DATASET_MAP = {'cora':CoraDataset}

def node_classification(embeddings, labels, test_size, **kwargs):
    result = dict()
    result['embedding'] = kwargs['name']
    result['micro_f1'] = list()
    result['macro_f1'] = list()
    
    for i in range(10):
        # split dataset
        train_X, test_X, train_Y, test_Y = train_test_split(
            embeddings, labels, test_size=test_size, random_state=i)
        
        if config.clf == 'nn':
            import torch
            from torch import nn
            from torch.nn import functional as F
            from torch.utils.data import DataLoader, Dataset
             
            in_feature = train_X.shape[1]
            number_of_classes = len(set(test_Y))

            one_hot_train_Y = torch.zeros(len(train_Y), number_of_classes).to(kwargs['device'])
            for idx, label in enumerate(train_Y):
                one_hot_train_Y[idx][label] = 1
            
            class TrainSet(Dataset):
                def __init__(self):
                    super(TrainSet, self).__init__()
                    self.train_X = torch.tensor(train_X).to(kwargs['device'])
                    self.train_Y = torch.tensor(train_Y).long().to(kwargs['device'])
                
                def __getitem__(self, idx):
                    return self.train_X[idx], self.train_Y[idx]
                
                def __len__(self):
                    return self.train_X.shape[0]
            
            class NeuralNet(nn.Module):
                def __init__(self):
                    super(NeuralNet, self).__init__()
                    self.fc1 = nn.Linear(in_feature, 500)
                    self.fc2 = nn.Linear(500, 100)
                    self.fc3 = nn.Linear(100, number_of_classes)
                    
                def forward(self, x):
                    x = F.relu(self.fc1(x))
                    x = F.relu(self.fc2(x))
                    x = self.fc3(x)
                    return x
                    
            clf = NeuralNet().to(kwargs['device'])
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.SGD(clf.parameters(), lr=0.001, momentum=0.9)
            
            for epoch in range(100):
                for x, y in DataLoader(TrainSet(), shuffle=True):
                   optimizer.zero_grad()
                   output = clf(x)
                   loss = criterion(output, y)
                   loss.backward()
                   optimizer.step()
            
            with torch.no_grad():
                pred = torch.max(clf(torch.tensor(test_X).to(kwargs['device'])), 1)[-1].cpu()
        
        elif config.clf == 'lr':
            clf = LogisticRegressionCV(cv=5, random_state=0, solver='lbfgs', 
                                       multi_class='multinomial', max_iter=1000000)
            clf.fit(train_X, train_Y)
            pred = clf.predict(test_X)
        
        result['micro_f1'].append(f1_score(pred, test_Y, average='micro'))
        result['macro_f1'].append(f1_score(pred, test_Y, average='macro'))
    
    result['micro_f1'] = sum(result['micro_f1'])/len(result['micro_f1'])
    result['macro_f1'] = sum(result['macro_f1'])/len(result['macro_f1'])
    
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
        if 'edgeprob' in model_tag:
            with torch.no_grad():
                model = torch.load(os.path.join(PICKLE_PATH, 'models', model_tag)).cuda()
                prob = model.forward1(torch.stack((z_src.cuda(), z_dst.cuda())).unsqueeze(0)).squeeze()
                pred.append(1 if float(prob) > 0.4 else 0)
        else:
            pred.append(1-cosine(z_src, z_dst)) 

 
    result = dict()
    result['model_tag'] = model_tag
    result['AUC'] = roc_auc_score(labels, pred)
    # AUC
    # True Positive Rate = 엣지 있는 pair들 중 몇 개나 엣지가 있다고 했는지.
    # False Positive Rate = 엣지 없는 pairs들 중 몇개나 엣지가 있다고 했는지.
    return result

def main(config):
    
    # load target embeddings. 
    target_paths = glob.glob(os.path.join(DATA_PATH, 
                                          'experiments', 'target',
                                          config.experiment, '*'))

    if config.experiment == 'node_classification':
        device_ = device('cuda:{}'.format(config.gpu))

        for target_path in target_paths:
            labels = DATASET_MAP[os.path.basename(target_path).split('_')[0]](device=device_).Y.numpy()
            try:
                embedding = pickle.load(open(target_path, 'rb'))
            except UnicodeDecodeError:
                embedding = pickle.load(open(target_path, 'rb'), encoding='latin-1')

            result = node_classification(embedding, labels, test_size=config.test_size,
                                         name=os.path.basename(target_path), device=device_) 
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
    
    # args for node classification
    parser.add_argument('--test_size', type=float)
    parser.add_argument('--clf', type=str)
    parser.add_argument('--gpu', type=int, default=0)

    config = parser.parse_args()
    print(config)
    main(config)

