import pdb
import os
import pickle
from multiprocessing.pool import ThreadPool

import requests
from PIL import Image
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3 import Retry
import numpy as np
from torch.nn.functional import normalize

from settings import PICKLE_PATH, DATA_PATH


def get(url):
    session = requests.Session()
    retries = Retry(total=5, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
    session.mount('https://', HTTPAdapter(max_retries=retries))
    response = session.get(url)

    return response


def edit_labels(negative_labels, positive_labels, dump=False):
    labels = pickle.load(open(os.path.join(PICKLE_PATH, 'wikipedia_labels'), 'rb'))

    for label in labels:
        for target in negative_labels + positive_labels:
            if target in label:
                if target in negative_labels:
                    label.remove(target)
                else:
                    label.add(target)

    if dump:
        pickle.dump(labels, open(os.path.join(PICKLE_PATH, 'wikipedia_labels'), 'wb'))
    return labels


def load_image(image_path, transform=None, max_size=None, shape=None):
    image = Image.open(image_path)

    if max_size:
        scale = max_size / max(image.size)
        size = np.array(image.size) * scale
        image = image.resize(size.astype(int), Image.ANTIALIAS)

    if shape:
        image = image.resize(shape, Image.LANCZOS)

    if transform:
        image = transform(image).unsqueeze_(0)

    return image


def get_labels(*target_labels, with_idx=False):
    default_labels = pickle.load(open(os.path.join(PICKLE_PATH, 'wikipedia_labels'), 'rb'))

    filtered_indices = []
    filtered_labels = []

    if len(target_labels) != 0:
        for idx, label in enumerate(default_labels):
            valid_labels = set(label).intersection(target_labels)
            if len(valid_labels) != 0:
                filtered_indices.append(idx)
                filtered_labels.append(label)
    else:
        filtered_labels = default_labels

    if with_idx:
        return filtered_labels, filtered_indices
    return filtered_labels


def get_docs():
    return pickle.load(open(os.path.join(PICKLE_PATH, 'wikipedia_docs'), 'rb'))

if __name__ == '__main__':
    docs = pickle.load(open(os.path.join(PICKLE_PATH, 'wikipedia_docs'), 'rb'))

    import nltk

    pool = ThreadPool(7)

    labels = [[] for i in range(len(docs))]

    def _painter_label(doc_idx):
        with open(os.path.join(DATA_PATH, 'wiki2vec', 'raw_html', str(doc_idx)), encoding='utf-8') as doc_io:
            soup = BeautifulSoup(doc_io.read(), 'html.parser')

        try:
            cats = soup.select_one('#mw-normal-catlinks').select('ul li a')
            cats = [cat.text.strip().lower() for cat in cats]
        except AttributeError:
            cats = []

        label = []

        for cat in cats:
            tokenized_cat = nltk.word_tokenize(cat)
            tagged_cat = nltk.pos_tag(tokenized_cat)
            pos = [idx for idx, tag in enumerate(tagged_cat) if tag[1] in ['IN', 'TO', 'VBG']]
            if len(pos) != 0:
                tokenized_cat = tokenized_cat[:pos[0]]

            if 'painters' in tokenized_cat:
                label.append('painter')
                label = list(set(label))

        return doc_idx, label

    for loop_idx, (doc_idx, label) in enumerate(pool.imap(_painter_label, range(len(docs)))):
        labels[doc_idx] = label
        print('[{}/{}]'.format(loop_idx, len(docs)), end='\r')

def normalize_elwise(*tensors):
    keep_dim = [tensor.shape for tensor in tensors]
    tensors = [tensor.flatten() for tensor in tensors]
    tensors = [normalize(tensor, dim=0) for tensor in tensors]
    tensors = [tensor.reshape(dim) for tensor, dim in zip(tensors, keep_dim)]
    return tuple(tensors)

