import argparse
import glob
import os
import pdb
import pickle

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import f1_score
from torch import device

from dataset import CoraDataset
from settings import DATA_PATH
from settings import PICKLE_PATH


def node_classification(embeddings, labels, **kwargs):
    # split dataset
    train_X, test_X, train_Y, test_Y = train_test_split(
        embeddings, labels, test_size=kwargs['test_size'], random_state=0)
    
    classifier = OneVsRestClassifier(LogisticRegression(
        random_state=0, solver='lbfgs', multi_class='multinomial', max_iter=300))

    classifier.fit(train_X, train_Y)
    score = classifier.score(test_X, test_Y)
    pred = classifier.predict(test_X)

    result = dict()
    result['embedding'] = kwargs['name']
    result['micro_f1'] = f1_score(pred, test_Y, average='micro')
    result['macro_f1'] = f1_score(pred, test_Y, average='macro')
    result['mean_accuracy'] = classifier.score(test_X, test_Y)
    return result

def main(config):
    device_ = device('cpu')
    
    # load target embeddings. 
    target_paths = glob.glob(os.path.join(DATA_PATH, 
                                          'experiments', 'target',
                                          config.experiment, '*'))

    embeddings = (pickle.load(open(target_path, 'rb')) for target_path in target_paths)
    if config.dataset == 'cora':
        labels = CoraDataset(device=device_).labels.numpy()
    else:
        raise ValueError
    
    if config.experiment == 'node_classification':
        for target_path in target_paths:
            try:
                embedding = pickle.load(open(target_path, 'rb'))
            except UnicodeDecodeError:
                embedding = pickle.load(open(target_path, 'rb'), encoding='latin-1')

            result = node_classification(embedding, labels, test_size=config.test_size,
                                         name=os.path.basename(target_path)) 
            print(result)
    else:
        raise ValueError


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--experiment', type=str, default='node_classification')
    parser.add_argument('--dataset', type=str, default='cora')
    parser.add_argument('--test_size', type=float, default='0.4')
    
    config = parser.parse_args()
    print(config)
    main(config)

