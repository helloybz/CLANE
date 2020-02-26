import argparse
import os

from sklearn.metrics import f1_score
import torch

from graph import Graph
from settings import PICKLE_PATH

parser = argparse.ArgumentParser()
parser.add_argument('--model_tag', type=str)
parser.add_argument('--gpu', type=int)
config = parser.parse_args()

dataset = config.model_tag.split('_')[0]
device = torch.device(f'cuda:{config.gpu}')

result = dict()
result['model_tag'] = config.model_tag

embeddings = torch.load(
        os.path.join(PICKLE_PATH, 'embeddings', config.model_tag),
        map_location=device
    )
model = torch.load(
        os.path.join(PICKLE_PATH, 'models', config.model_tag),
        map_location=device
    )
original_graph = Graph(dataset, True, 1433, device)

mean, std = embeddings.mean(), embeddings.std()
embeddings = (embeddings-mean)/std

reconstructed_A = model(embeddings, embeddings)
original_A = original_graph.A
result['macro_f1'] = list()
for i in range(10):
    result['macro_f1'].append(
            f1_score(
                original_A.flatten().cpu(),
                reconstructed_A.bernoulli().detach().cpu(),
                average='macro'
            ))

result['macro_f1'] = sum(result['macro_f1'])/10
for key in result.keys():
    print(f'{key}: {result[key]}')

