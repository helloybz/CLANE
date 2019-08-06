import argparse
import os

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import torch

from graph import Graph
from settings import PICKLE_PATH


parser = argparse.ArgumentParser()
parser.add_argument('--model_tag', type=str)
parser.add_argument('--test_size', type=float)
config = parser.parse_args()

# load data
embedding = torch.load(os.path.join(PICKLE_PATH, 'embeddings', config.model_tag), map_location=torch.device('cpu')).numpy()
dataset = config.model_tag.split('_')[0]
labels = Graph(dataset, True, 256, torch.device('cpu')).Y.numpy()
result = dict()
result['embedding'] = config.model_tag
result['micro_f1'] = list()
result['macro_f1'] = list()

for _ in range(30):
    # split data
    train_X, test_X, train_Y, test_Y = train_test_split(embedding, labels, test_size=config.test_size)

    # make classifier
    classifier = LogisticRegression(solver='lbfgs', multi_class='ovr', max_iter=1e+6, n_jobs=4)

    # train classifier
    classifier.fit(train_X, train_Y)
    pred = classifier.predict(test_X)

    # log scores
    result['micro_f1'].append(f1_score(pred, test_Y, average='micro'))
    result['macro_f1'].append(f1_score(pred, test_Y, average='macro'))

result['micro_f1'] = sum(result['micro_f1'])/30
result['macro_f1'] = sum(result['macro_f1'])/30

# print results
for key in result.keys():
    print(f'{key:9}: {result[key]}')
