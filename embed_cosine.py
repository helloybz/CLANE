import argparse
import os
import pdb
import pickle

from numpy import inf
from sklearn.metrics.pairwise import cosine_similarity
import torch
from torch.nn import functional as F

from dataset import CoraDataset, CiteseerDataset
from settings import PICKLE_PATH
from embed import update_embedding


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--gamma', type=float, default=0.74)
parser.add_argument('--tolerence_Z', type=int, default=30)
parser.add_argument('--model_tag', type=str)
config = parser.parse_args()



if __name__ == "__main__":
    device = torch.device(f'cuda:{config.gpu}') if config.gpu is not None else torch.device('cpu')
    gamma = config.gamma
    tolerence_Z = config.tolerence_Z

    if config.dataset == 'cora':
        graph = CoraDataset(device=device)
    else:
        raise ValueError

    import pdb; pdb.set_trace() 
    sim_metric = lambda x,y: F.cosine_similarity(x,y,dim=-1)

    min_distance = inf
    tolerence = tolerence_Z
    recent_Z = graph.Z.clone()
    while tolerence != 0:
        distance = update_embedding(
                graph, 
                sim_metric,
                recent_Z,
                gamma
            )
        if min_distance > distance:
            min_distance = distance
            tolerence = tolerence_Z
        else:
            tolerence -= 1
        
        print(distance,end='\r')
    pickle.dump(graph.Z.cpu().numpy(), open(os.path.join(PICKLE_PATH, 'embedding', config.model_tag),'wb'))
    
