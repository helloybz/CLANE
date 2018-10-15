import argparse
import glob
import os
import pickle
from multiprocessing.pool import ThreadPool

import numpy as np
from gensim.parsing.preprocessing import preprocess_string
from scipy.spatial.distance import euclidean, cosine
from torch.nn import functional as F
import torch

from data import checkup_images
from helper import load_image
from models import device, Resnet18, TEXT_FILTERS, Doc2Vec, Resnet152
from settings import DATA_PATH, BASE_DIR
from settings import PICKLE_PATH


# def load_image(image_path, transform=None, max_size=None, shape=None):
#     image = Image.open(image_path)
#
#     if max_size:
#         scale = max_size / max(image.size)
#         size = np.array(image.size) * scale
#         image = image.resize(size.astype(int), Image.ANTIALIAS)
#
#     if shape:
#         image = image.resize(shape, Image.LANCZOS)
#
#     if transform:
#         image = transform(image).unsqueeze_(0)
#
#     return image.to(device)


def get_content_vectors(dataset):
    if dataset == 'wikipedia':
        # Load or Calculate C_X
        if not os.path.exists(os.path.join(PICKLE_PATH, 'wikipedia_c_{0}_{1}'.format(config.img_embedder, config.doc2vec_size))):

            # Image Modality Model
            if config.img_embedder == 'resnet18':
                img_embedder = Resnet18().to(device)
            elif config.img_embedder == 'resnet152':
                img_embedder = Resnet152().to(device)
            else:
                raise argparse.ArgumentError
            img_embedder.eval()
            print('set model to eval mode')

            # Text Modality Model
            if os.path.exists(os.path.join(PICKLE_PATH, 'wikipedia_doc2vec_embedder_{0}'.format(config.doc2vec_size))):
                txt_embedder = pickle.load(open(os.path.join(PICKLE_PATH, 'wikipedia_doc2vec_embedder_{0}'.format(config.doc2vec_size)), 'rb'))
            else:
                doc_paths = glob.glob(os.path.join(DATA_PATH, 'wiki2vec', 'full_text', '*'))
                txts = [open(path, 'r', encoding='utf-8').read() for path in doc_paths]
                txts = [preprocess_string(txt, TEXT_FILTERS) for txt in txts]
                txt_embedder = Doc2Vec(txts, vector_size=config.doc2vec_size)
                pickle.dump(txt_embedder, open(os.path.join(PICKLE_PATH, 'wikipedia_doc2vec_embedder_{0}'.format(config.doc2vec_size)), 'wb'))
            print('Text modal model prepared.')

            docs = pickle.load(open(os.path.join(PICKLE_PATH, 'wikipedia_docs'), 'rb'))

            pool = ThreadPool(5)

            c = None

            for idx, (doc_id, doc_title) in enumerate(pool.imap(lambda x: x, enumerate(docs))):
                while True:
                    img_paths = glob.glob(os.path.join(DATA_PATH, 'wiki2vec', 'images', str(doc_id) + '_*'))
                    img_set = list()
                    for path in img_paths:
                        try:
                            img = load_image(path, transform=img_embedder.transform, shape=(224, 224)).to(device)
                            img_set.append(img)
                        except Exception:
                            with open(os.path.join(BASE_DIR, 'bad_images'), 'a') as bad_img_io:
                                bad_img_io.write(os.path.basename(path))

                    with open(os.path.join(DATA_PATH, 'wiki2vec', 'full_text', str(doc_id)), 'r',
                              encoding='utf-8') as txt_io:
                        text = txt_io.read()
                    try:
                        img_feature = img_embedder(img_set)
                        break
                    except FileNotFoundError:
                        labels = pickle.load(open(os.path.join(PICKLE_PATH, 'wikipedia_labels'), 'rb'))
                        checkup_images(doc_idx=doc_id, doc_title=doc_title, doc_labels=labels[doc_id])

                text_feature = txt_embedder.forward(text)
                merged_feature = np.hstack((F.normalize(torch.Tensor(text_feature)), F.normalize(img_feature)))

                if c is not None:
                    c = np.vstack((c, merged_feature))
                else:
                    c = merged_feature

                print('Calculating C done {0:%}'.format(idx / len(docs)), end="\r")

            pool.close()

            pickle.dump(c, open(
                os.path.join(PICKLE_PATH, 'wikipedia_c_{0}_{1}'.format(config.img_embedder, config.doc2vec_size)),
                'wb'))
        else:
            c = pickle.load(
                open(os.path.join(PICKLE_PATH,
                                  'wikipedia_c_{0}_{1}'.format(config.img_embedder, config.doc2vec_size)),
                     'rb'))

        return c

    elif dataset == 'citation':
        print('Preparing citaitions abstracts')
        abstracts = pickle.load(open(os.path.join(PICKLE_PATH, 'citation_abstracts'), 'rb'))
        abstracts = [preprocess_string(abstract, TEXT_FILTERS) for abstract in abstracts]

        print('Preparing txt_embedder.')
        if not os.path.exists(os.path.join(PICKLE_PATH, 'citation_doc2vec_embedder')):
            txt_embedder = Doc2Vec(abstracts, vector_size=config.doc2vec_size)
            pickle.dump(txt_embedder, open(os.path.join(PICKLE_PATH, 'citation_doc2vec_embedder'), 'wb'))
        else:
            txt_embedder = pickle.load(open(os.path.join(PICKLE_PATH, 'citation_doc2vec_embedder'), 'rb'))

        c = None
        abstracts = pickle.load(open(os.path.join(PICKLE_PATH, 'citation_abstracts'), 'rb'))
        print('Preparing content vector.')
        for idx, abstract in enumerate(abstracts):
            text_feature = txt_embedder.forward(abstract)
            if c is not None:
                c = np.vstack((c, text_feature))
            else:
                c = text_feature

            print('Calculating C done {0:%}'.format(idx / len(abstracts)), end="\r")
    return


def get_similarity_vectors(dataset, **kwargs):
    if dataset == 'wikipedia':
        if not os.path.exists(
                os.path.join(PICKLE_PATH, 'wikipedia_s_{0}_{1}'.format(config.img_embedder, config.doc2vec_size))):
            if 'c' not in kwargs.keys():
                c = pickle.load(
                    open(os.path.join(PICKLE_PATH,
                                      'wikipedia_c_{0}_{1}'.format(config.img_embedder, config.doc2vec_size)),
                         'rb'))
            else:
                c = kwargs['c']

            if 'edges' not in kwargs.keys():
                edges = pickle.load(
                    open(os.path.join(PICKLE_PATH, '{0}_edges'.format(dataset)),
                         'rb'))
            else:
                edges = kwargs['edges']

            print('Calculating s.')
            for edge in edges:
                print(c[edge[0]])
                print(c[edge[1]])
                break
        else:
            print('Loading s from file.')
            s = pickle.load(open(os.path.join(PICKLE_PATH, 'wikipedia_s_{0}_{1}'.format(config.img_embedder, config.doc2vec_size)), 'rb'))
            print('Loading s from file done.')

        return s


def main(config):
    c = get_content_vectors(dataset=config.dataset)

    v = c.copy()

    edges = pickle.load(open(os.path.join(PICKLE_PATH, '{0}_edges'.format(config.dataset)), 'rb'))

    # if not os.path.exists(os.path.join(PICKLE_PATH, '{0}_s_{1}'.format(config.dataset, config.sim_metric))):
    #     s = {}
    #     pickle.dump(s, open(os.path.join(PICKLE_PATH, '{0}_s_{1}'.format(config.dataset, config.sim_metric)), 'wb'))
    # else:
    #     s = pickle.load(open(os.path.join(PICKLE_PATH, '{0}_s_{1}'.format(config.dataset, config.sim_metric)), 'rb'))

    # Initialize Similarity matrix

    is_converged = False

    def get_similarity(index_1, index_2):
        if config.sim_metric == 'cosine_C':
            return 1 - cosine(c[index_1], c[index_2])

    def update_v(args):
        node_id = args[0]
        c_x = args[1]

        ref_ids = [edge[1] for edge in edges if edge[0] == node_id]
        sims = [np.exp(get_similarity(node_id, ref_id)) for ref_id in ref_ids]
        regularized_sims = [sim / sum(sims) for sim in sims]
        diff = config.alpha * sum([r_sim * v[ref_id] for ref_id, r_sim in zip(ref_ids, regularized_sims)])

        delta = euclidean(v[node_id], c_x + diff)

        v[node_id] = c_x + diff

        # pdb.set_trace()
        return delta > config.threshold, delta

    pool = ThreadPool(1)

    iteration_counter = 0

    while not is_converged:
        is_converged = True
        max_delta = -np.inf
        for idx, results in enumerate(pool.imap(update_v, enumerate(c))):
            max_delta = max(max_delta, results[1])
            print('iter{0}  {1:%} done. delta: {2}'.format(iteration_counter, (idx / len(c)), max_delta), end='\r')
        is_converged = max_delta < config.threshold
        iteration_counter += 1
    pool.close()

    pickle.dump(v, open(os.path.join(PICKLE_PATH, '{}_v_{}_txt{}_thr{}_a{}'.format(
        config.dataset,
        config.img_embedder,
        config.doc2vec_size,
        config.sim_metric,
        config.alpha,
        config.threshold,
    )), 'wb'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='wikipedia')
    parser.add_argument('--img_embedder', type=str, default='resnet18')
    parser.add_argument('--doc2vec_size', type=int, default=1024)
    parser.add_argument('--sim_metric', type=str, default='cosine_C')
    parser.add_argument('--alpha', type=float, default=0.9)
    parser.add_argument('--threshold', type=float, default=0.01)
    config = parser.parse_args()
    print(config)
    main(config)
