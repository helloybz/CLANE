import argparse
import os
import pickle

from sklearn.decomposition import SparsePCA

from settings import DATA_PATH, PICKLE_PATH


parser = argparse.ArgumentParser()
parser.add_argument('--src', type=str)
parser.add_argument('--dim', type=int)
config=parser.parse_args()

raw_embeddings = pickle.load(
        open(os.path.join(PICKLE_PATH, 'embedding', config.src), 'rb'))
pca = SparsePCA(
        n_components=config.dim,
        normalize_components=True,
        n_jobs=-1,
        random_state=0
    )
new_embeddings = pca.fit_transform(raw_embeddings)
import torch



pickle.dump(
        torch.FloatTensor(new_embeddings).numpy(), 
        open(os.path.join(PICKLE_PATH, 'embedding', f'{config.src}_d{config.dim}'),'wb'))

