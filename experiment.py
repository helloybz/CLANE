import argparse
import glob
import os
import pickle

from scipy.spatial.distance import cosine
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from torch import device
import torch

from dataset import CoraDataset
from settings import DATA_PATH
from settings import PICKLE_PATH

DATASET_MAP = {
    'cora':CoraDataset,
    }


def node_classification(embeddings, labels, test_size, name, **kwargs):
    result = dict()
    result['embedding'] = name
    result['micro_f1'] = list()
    result['macro_f1'] = list()
    
    # split dataset
    for i in range(10):
        train_X, test_X, train_Y, test_Y = train_test_split(embeddings, labels, test_size=test_size)
          
        classifier = LogisticRegression(solver='lbfgs', multi_class='multinomial', max_iter=1000000, n_jobs=4)
        classifier.fit(train_X, train_Y)
        pred = classifier.predict(test_X)

        result['micro_f1'].append(f1_score(pred, test_Y, average='micro'))
        result['macro_f1'].append(f1_score(pred, test_Y, average='macro'))

    result['micro_f1'] = sum(result['micro_f1'])/10
    result['macro_f1'] = sum(result['macro_f1'])/10
    
    return result

def link_prediction(model_tag, **kwargs):
    target_embedding = pickle.load(
            open(os.path.join(DATA_PATH, 'experiments', 'target', 'link_prediction', model_tag), 'rb'))
    target_network = DATASET_MAP[model_tag.split('_')[0]](deepwalk=False, device=kwargs['device'], sampled=True)
    target_network.Z = torch.tensor(target_embedding).to(kwargs['device'])
    original_network = DATASET_MAP[model_tag.split('_')[0]](deepwalk=False, device=kwargs['device'])
    
    removed_edges = (original_network.A != target_network.A).nonzero()
    from random import sample
    nonedges = (original_network.A==0).nonzero()
    non_edges_idx = sample(range(nonedges.shape[0]), k=removed_edges.shape[0])
    negative_edges = nonedges[non_edges_idx]

    test_edges = torch.cat([removed_edges, negative_edges])
    labels = [1]*removed_edges.shape[0] + [0]*negative_edges.shape[0]
    
    preds = list()

    if 'edgeprob' in model_tag:
        model = torch.load(os.path.join(PICKLE_PATH, 'models', model_tag)).to(kwargs['device'])

    for src, dst in test_edges: 
        z_src = target_network.Z[src]
        z_dst = target_network.Z[dst]
#        if 'edgeprob' in model_tag:
#            with torch.no_grad():
#                pred = float(model(z_src, z_dst.unsqueeze(0)))
#        else:
        pred = float(torch.dot(z_src, z_dst).sigmoid())

        preds.append(pred)
    preds = torch.tensor(preds)
    _, idx = preds.sort(descending=True)

    presicion_at_k = (idx[:removed_edges.shape[0]]<removed_edges.shape[0]).float().mean()
    result = dict()
    result['model_tag'] = model_tag
    result['AUC'] = roc_auc_score(labels, preds)
    result['presicion_at_{}'.format(removed_edges.shape[0])] = presicion_at_k
    # AUC
    # True Positive Rate = 엣지 있는 pair들 중 몇 개나 엣지가 있다고 했는지.
    # False Positive Rate = 엣지 없는 pairs들 중 몇개나 엣지가 있다고 했는지.
    return result

@torch.no_grad()
def graph_reconstruction(embeddings, sim_metric, original_A, ks, **kwargs):
    import pdb; pdb.set_trace()
    result = dict()
    result['model_tag'] = kwargs['model_tag']
    embeddings = torch.tensor(embeddings, device=kwargs['device'])
    
    reconstructed_A = torch.zeros(original_A.shape)
    for src in range(embeddings.shape[0]):
        sim = sim_metric(embeddings[src], embeddings)
        reconstructed_A[src] = sim
    
    _, indices = reconstructed_A.flatten().sort(descending=True)
  
    
    for k in ks:
        result[f'precision@{k}'] = original_A.flatten()[indices][:k].sum().div(k).item()

    return result


def main(config):
    
    # load target embeddings. 
    target_paths = glob.glob(os.path.join(DATA_PATH, 
                                          'experiments', 'target',
                                          config.experiment, '*'))

    if config.experiment == 'node_classification':
        for target_path in target_paths:
            labels = DATASET_MAP[os.path.basename(target_path).split('_')[0]]().Y.cpu().numpy()
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
            device = torch.device('cuda:{}'.format(config.gpu))
            result = link_prediction(model_tag=model_tag, device=device)
       
            print('===========================')
            for key in result.keys():
                if key == 'model_tag':
                    print(result[key])
                else:
                    print(key, '\t', '{:4f}'.format(result[key]))

    elif config.experiment == 'reconstruction':

        if config.gpu is not None:
            device = torch.device(f'cuda:{config.gpu}')
        else:
            device =  torch.device('cpu')
        
        for target_path in target_paths:
            model_tag = os.path.basename(target_path)
            if 'edgeprob' in target_path:
                sim_metric = torch.load(os.path.join(PICKLE_PATH, 'models', model_tag), map_location=device)
            else:
                sim_metric = lambda x,y: torch.nn.functional.cosine_similarity(x,y,dim=-1)

            if 'cora' in target_path:
                original_A = CoraDataset(device=device).A
            else:
                raise ValueError

            result = graph_reconstruction(
                embeddings=pickle.load(open(target_path, 'rb')),
                sim_metric=sim_metric,
                original_A=original_A,
                ks=config.k,
                model_tag=model_tag,
                device=device
            )
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
    parser.add_argument('--gpu', type=str)
    parser.add_argument('--k', nargs='+', type=int)

    config = parser.parse_args()
    print(config)
    main(config)

