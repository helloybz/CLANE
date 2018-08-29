import argparse
import glob
import os
import pickle
import re
from multiprocessing.pool import ThreadPool

import numpy as np
from PIL import Image
from gensim.parsing.preprocessing import preprocess_string
from scipy.spatial.distance import pdist, euclidean, squareform

from models import device, Resnet18, TEXT_FILTERS, Doc2Vec
from settings import DATA_PATH, BASE_DIR
from settings import PICKLE_PATH


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

    return image.to(device)


def get_content_vectors(dataset):
    if dataset == 'wikipedia':
        # Load or Calculate C_X
        if not os.path.exists(
                os.path.join(PICKLE_PATH, 'wikipedia_c_{0}_{1}'.format(config.img_embedder, config.doc2vec_size))):
            # Image Modality Model
            if config.img_embedder == 'resnet18':
                img_embedder = Resnet18().to(device)
            else:
                raise argparse.ArgumentError

            # Text Modality Model
            if os.path.exists(os.path.join(PICKLE_PATH, 'wikipedia_doc2vec_embedder')):
                txt_embedder = pickle.load(open(os.path.join(PICKLE_PATH, 'wikipedia_doc2vec_embedder'), 'rb'))
            else:
                doc_paths = glob.glob(os.path.join(DATA_PATH, 'wiki2vec', 'full_text', '*'))
                txts = [open(path, 'r', encoding='utf-8').read() for path in doc_paths]
                txts = [preprocess_string(txt, TEXT_FILTERS) for txt in txts]
                txt_embedder = Doc2Vec(txts, vector_size=config.doc2vec_size)
                pickle.dump(txt_embedder, open(os.path.join(PICKLE_PATH, 'wikipedia_doc2vec_embedder'), 'wb'))
            docs = pickle.load(open(os.path.join(PICKLE_PATH, 'wikipedia_docs'), 'rb'))

            pool = ThreadPool(5)

            c = None

            for idx, doc_title in enumerate(pool.imap(lambda x: x, docs)):
                img_paths = glob.glob(os.path.join(DATA_PATH, 'wiki2vec', 'images', str(idx) + '_*'))
                img_set = list()
                for path in img_paths:
                    try:
                        img = load_image(path, transform=img_embedder.transform, shape=(255, 255))
                        img_set.append(img)
                    except Exception:
                        with open(os.path.join(BASE_DIR, 'bad_images'), 'a') as bad_img_io:
                            bad_img_io.write(os.path.basename(path))

                with open(os.path.join(DATA_PATH, 'wiki2vec', 'full_text', str(idx)), 'r', encoding='utf-8') as txt_io:
                    text = txt_io.read()

                img_feature = img_embedder.forward(img_set)
                text_feature = txt_embedder.forward(text)
                merged_feature = np.hstack((text_feature, img_feature))

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
                open(os.path.join(PICKLE_PATH, 'wikipedia_c_{0}_{1}'.format(config.img_embedder, config.doc2vec_size)),
                     'rb'))

        return c
    elif dataset == 'citation':
        with open(os.path.join(DATA_PATH, 'citation-network', 'outputacm.txt'), 'r', encoding='utf-8') as test_io:
            citations = test_io.read()

        citations = citations.split('\n\n')
        citations = [citation for citation in citations if '#!' in citation]

        parsed_citations = dict()
        abstracts = list()
        indexes = list()
        edges = list()
        venues = list()
        for idx, citation in enumerate(citations):
            tokens = re.split(r'(#\*|#@|#!|#t|#c|#index|#%)', citation)
            abstract = tokens[tokens.index('#!') + 1].strip()
            venue = tokens[tokens.index('#c') + 1].strip()
            ref_ids = [int(tokens[idx + 1].strip()) for idx, token in enumerate(tokens) if token == '#%']
            if venue.strip() == '':
                continue
            index = int(tokens[tokens.index('#index') + 1])

            abstracts.append(abstract)
            indexes.append(index)
            venues.append(venue)
            for ref_id in ref_ids:
                edges.append((index, ref_id))

        # print(abstracts)
        # print(indexes)
        # print(venues)
        # print(edges)
        abstracts = [preprocess_string(abstract, TEXT_FILTERS) for abstract in abstracts]
        txt_embedder = Doc2Vec(abstracts, vector_size=config.doc2vec_size)
        pickle.dump(txt_embedder, open(os.path.join(PICKLE_PATH, 'citations_embedder')))
    return


def get_similarity_vectors(dataset):
    if dataset == 'wikipedia':
        if not os.path.exists(
                os.path.join(PICKLE_PATH, 'wikipedia_s_{0}_{1}'.format(config.img_embedder, config.doc2vec_size))):
            c = pickle.load(
                open(os.path.join(PICKLE_PATH, 'wikipedia_c_{0}_{1}'.format(config.img_embedder, config.doc2vec_size)),
                     'rb'))

            print('Calculating s.')
            s = 1 - squareform(pdist(c, metric='cosine'))
            print('Writing s into file.')
            pickle.dump(s, open(os.path.join(PICKLE_PATH, 'wikipedia_s_{0}_{1}'.format(config.img_embedder, config.doc2vec_size)), 'wb'), protocol=4)
            print('Writing s into file done.')
        else:
            print('Loading s from file.')
            s = pickle.load(open(os.path.join(PICKLE_PATH, 'wikipedia_s_{0}_{1}'.format(config.img_embedder, config.doc2vec_size)), 'rb'))
            print('Loading s from file done.')

        return s
    elif dataset == 'citation':
        pass


def main(config):
    """Get Content Vectors"""
    c = get_content_vectors(dataset=config.dataset)

    v = c.copy()

    s = get_similarity_vectors(dataset=config.dataset)

    edges = pickle.load(open(os.path.join(PICKLE_PATH, '{0}_edges'.format(config.dataset)), 'rb'))
    # Initialize Similarity matrix

    is_converged = False

    def update_v(args):
        node_id = args[0]
        c_x = args[1]

        ref_ids = [edge[1] for edge in edges if edge[0] == node_id]
        sims = [np.exp(s[node_id][ref_id]) for ref_id in ref_ids]
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

    pickle.dump(v, open(os.path.join(PICKLE_PATH, 'wikipedia_v_thr{0}'.format(config.threshold)), 'wb'))
#

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='wikipedia')
    parser.add_argument('--img_embedder', type=str, default='resnet18')
    parser.add_argument('--doc2vec_size', type=int, default=1024)
    parser.add_argument('--alpha', type=float, default=0.9)
    parser.add_argument('--threshold', type=float, default=0.01)
    config = parser.parse_args()
    print(config)
    main(config)
