import argparse
import glob
import os
import pickle
import time

import gensim
import numpy as np
import torch
import torchvision.models as models
from PIL import Image
from gensim.parsing.preprocessing import strip_multiple_whitespaces
from gensim.parsing.preprocessing import strip_non_alphanum, preprocess_string
from gensim.parsing.preprocessing import strip_numeric
from gensim.parsing.preprocessing import strip_short
from torch import nn
from torch.nn.functional import cosine_similarity
from torchvision import transforms

from models import device, Resnet18, TEXT_FILTERS, Doc2Vec
from settings import DATA_PATH
from settings import PICKLE_PATH


def _load_images(image_path, transform=None, max_size=None, shape=None):
    image = Image.open(image_path)

    if max_size:
        scale = max_size / max(image.size)
        size = np.array(image.size) * scale
        image = image.resize(size.astype(int), Image.ANTIALIAS)

    if shape:
        image = image.resize(shape, Image.LANCZOS)

    if transform:
        image = transform(image).unsqueeze_(0)

    return image.to(device)


def main(config):
    with open(os.path.join(PICKLE_PATH, 'doc_ids'), 'rb') as doc_mapping_io:
        doc_ids = pickle.load(doc_mapping_io)

    # process config args
    if config.img_embedder == 'resnet18':
        img_embedder = Resnet18().to(device)

    if config.doc2vec_text == 'abstract':
        doc2vec_pickle_name = 'doc2vec_abstract'
    elif config.doc2vec_text == 'full_text':
        doc2vec_pickle_name = 'doc2vec_full_text'
    else:
        raise argparse.ArgumentError

    #
    if os.path.exists(os.path.join(PICKLE_PATH, doc2vec_pickle_name)):
        txt_embedder = pickle.load(open(os.path.join(PICKLE_PATH, doc2vec_pickle_name), 'rb'))
    else:
        doc_paths = glob.glob(os.path.join(DATA_PATH, config.doc2vec_text, '*'))
        txts = [open(path, 'r', encoding='utf-8').read() for path in doc_paths]
        txts = [preprocess_string(txt, TEXT_FILTERS) for txt in txts]
        txt_embedder = Doc2Vec(txts, vector_size=config.doc2vec_size)
        pickle.dump(txt_embedder, open(os.path.join(PICKLE_PATH, doc2vec_pickle_name), 'wb'))

    resnet18_transform = transforms.Compose([
        transforms.Lambda(lambda x: x.convert('RGB')),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.size()[0] == 1 else x)
    ])

    if not os.path.exists(os.path.join(PICKLE_PATH, 'c_x')):
        c_x = None
        text_paths = glob.glob(os.path.join(DATA_PATH, config.doc2vec_text, '*'))
        for idx, text_path in enumerate(text_paths):
            try:
                doc_id = os.path.basename(text_path).split('_')[-1]
                img_set = glob.glob(os.path.join(DATA_PATH, 'images', 'image_' + doc_id + '_*'))
                img_set = [_load_images(path, transform=resnet18_transform, shape=(255, 255)) for path in img_set]

                with open(text_path, 'r', encoding='utf-8') as txt_io:
                    txt_feature = txt_embedder.forward(txt_io.read())
                img_feature = img_embedder.forward(img_set)
                merged_feature = np.hstack((txt_feature, img_feature))
                if c_x is not None:
                    c_x = np.vstack((c_x, merged_feature))
                else:
                    c_x = merged_feature
                print('done {0:%} {1:}'.format(idx / len(text_paths), c_x.shape), end="\r")
            except Exception:
                doc_list = pickle.load(open(os.path.join(PICKLE_PATH, 'doc_name_list'), 'rb'))
                doc_id = int(os.path.basename(text_path).split('_')[-1])
                print(text_path, doc_list[doc_id])
                raise Exception

        pickle.dump(c_x, open(os.path.join(PICKLE_PATH, 'c_x'), 'wb'))

    else:
        c_x = pickle.load(open(os.path.join(PICKLE_PATH, 'c_x'), 'rb'))

    v_x = c_x
    del c_x

    # update v_x
    network = pickle.load(open(os.path.join(PICKLE_PATH, 'network'), 'rb'))

    ## Initialize Similarity matrix
    for node_id, doc in enumerate(doc_ids):
        v = v_x[node_id]
        reference_ids = network[node_id]
        for reference_id in reference_ids:
            cosine_similarity(v, v_x[reference_id])


# Calculate Similarity


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_embedder', type=str, default='resnet18')
    parser.add_argument('--doc2vec_size', type=int, default=2048)
    parser.add_argument('--doc2vec_text', type=str, default='abstract')
    config = parser.parse_args()
    print(config)
    main(config)
