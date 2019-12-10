import argparse
import os

from sklearn.cluster import KMeans
from sklearn.metrics import homogeneity_score
from sklearn.metrics import silhouette_score
import torch

from graph import Graph 
from settings import PICKLE_PATH


parser = argparse.ArgumentParser()
parser.add_argument('--model_tag', type=str)
parser.add_argument('--n_cluster', type=int)
config = parser.parse_args()

# Load the embedding
embedding = torch.load(os.path.join(PICKLE_PATH, 'embeddings', config.model_tag), map_location=torch.device('cpu'))
dataset = config.model_tag.split('_')[0]
labels = Graph(dataset, True, 245, torch.device('cpu')).Y.numpy()
result = dict()
result['embedding'] = config.model_tag
 
clustering = KMeans(config.n_cluster, n_jobs=-1)
clustering.fit(embedding)

result['homogeneity'] = homogeneity_score(labels, clustering.labels_)
result['silhouette'] = silhouette_score(embedding, clustering.labels_)

for key in result.keys():
    print(f'{key}: {result[key]}')
