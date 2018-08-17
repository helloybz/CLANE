import argparse
import glob
import os
import pickle
from multiprocessing.pool import ThreadPool
import pdb
import numpy as np
from PIL import Image
from gensim.parsing.preprocessing import preprocess_string
from scipy.spatial import distance

from data import checkup_images
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

    # Image Modality Model
    if config.img_embedder == 'resnet18':
        img_embedder = Resnet18().to(device)
    else:
        raise argparse.ArgumentError

    if os.path.exists(os.path.join(PICKLE_PATH, 'doc2vec_embedder')):
        txt_embedder = pickle.load(open(os.path.join(PICKLE_PATH, 'doc2vec_embedder'), 'rb'))
    else:
        doc_paths = glob.glob(os.path.join(DATA_PATH, 'full_text', '*'))
        txts = [open(path, 'r', encoding='utf-8').read() for path in doc_paths]
        txts = [preprocess_string(txt, TEXT_FILTERS) for txt in txts]
        txt_embedder = Doc2Vec(txts, vector_size=config.doc2vec_size)
        pickle.dump(txt_embedder, open(os.path.join(PICKLE_PATH, 'doc2vec_embedder'), 'wb'))

    # Load or Calculate C_X
    if not os.path.exists(os.path.join(PICKLE_PATH, 'c_x_{0}_{1}'.format(config.img_embedder, config.doc2vec_size))):
        c_list = None
        if not os.path.exists(os.path.join(PICKLE_PATH, 'ids')):
            ids = [os.path.basename(path) for path in glob.glob(os.path.join(DATA_PATH, 'full_text', '*'))]
            pickle.dump(ids, open(os.path.join(PICKLE_PATH, 'ids'), 'wb'))
        else:
            ids = pickle.load(open(os.path.join(PICKLE_PATH, 'ids'), 'rb'))

        for idx, doc_id in enumerate(ids):
            img_set = glob.glob(os.path.join(DATA_PATH, 'images', doc_id + '_*'))
            while True:
                try:
                    img_set = [_load_images(path, transform=img_embedder.transform, shape=(255, 255)) for path in img_set]
                    break
                except Exception:
                    print('Image load error.')
                    checkup_images(doc_id)
                    pass

            with open(os.path.join(DATA_PATH, 'full_text', doc_id), 'r', encoding='utf-8') as txt_io:
                txt_feature = txt_embedder.forward(txt_io.read())
            img_feature = img_embedder.forward(img_set)
            merged_feature = np.hstack((txt_feature, img_feature))
            if c_list is not None:
                c_list = np.vstack((c_list, merged_feature))
            else:
                c_list = merged_feature
            print('Calculating C_X done {0:%} {1:}'.format(idx / len(ids), c_list.shape), end="\r")

        pickle.dump(c_list, open(os.path.join(PICKLE_PATH, 'c_x_{0}_{1}'.format(config.img_embedder, config.doc2vec_size)), 'wb'))

    else:
        c_list = pickle.load(open(os.path.join(PICKLE_PATH, 'c_x_{0}_{1}'.format(config.img_embedder, config.doc2vec_size)), 'rb'))

    # v_list = c_list.copy()
    #
    # # update v_matrix
    # network = pickle.load(open(os.path.join(PICKLE_PATH, 'network'), 'rb'))
    # # Initialize Similarity matrix
    # s_c_x = pickle.load(open(os.path.join(PICKLE_PATH, 's_c_x'), 'rb'))
    #
    # is_converged = False
    #
    # def update_v(args):
    #     node_id = args[0]
    #     c_x = c_list[node_id]
    #
    #     ref_ids = network[str(node_id)]
    #     sims = [np.exp(s_c_x[node_id][ref_id]) for ref_id in ref_ids]
    #     regularized_sims = [sim / sum(sims) for sim in sims]
    #     diff = config.alpha * sum([r_sim * v_list[ref_id] for ref_id, r_sim in zip(ref_ids, regularized_sims)])
    #
    #     delta = distance.euclidean(v_list[node_id], c_x + diff)
    #
    #     v_list[node_id] = c_x + diff
    #
    #     # pdb.set_trace()
    #     return delta > config.threshold, delta
    #
    # pool = ThreadPool(1)
    #
    # iteration_counter = 0
    #
    # while not is_converged:
    #     is_converged = True
    #     max_delta = -np.inf
    #     for idx, results in enumerate(pool.imap(update_v, enumerate(doc_ids))):
    #         # if results[0]:
    #         #	is_converged = results[0]
    #         max_delta = max(max_delta, results[1])
    #         print('iter{0}  {1:%} done. delta: {2}'.format(iteration_counter, (idx / len(doc_ids)), max_delta),
    #               end='\r')
    #     is_converged = max_delta < config.threshold
    #     iteration_counter += 1
    # pool.close()
    #
    # pickle.dump(v_list, open(os.path.join(PICKLE_PATH, 'v_list_thr{0}'.format(config.threshold)), 'wb'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_embedder', type=str, default='resnet18')
    parser.add_argument('--doc2vec_size', type=int, default=1024)
    parser.add_argument('--alpha', type=float, default=0.9)
    parser.add_argument('--threshold', type=float, default=0.1)
    config = parser.parse_args()
    print(config)
    main(config)
