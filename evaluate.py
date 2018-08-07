"""
Evaluate

1. Clustering
"""
import argparse
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.decomposition import PCA

from settings import PICKLE_PATH, BASE_DIR, WIKIPEIDA_CATEGORIES

# local helpers
COLOR_MAP = ['red', 'blue', 'green']


def _load_embeding(pickle_name):
    return pickle.load(open(os.path.join(PICKLE_PATH, pickle_name), 'rb'))


# dimension reductions
def dim_reduc_pca(emb):
    model = PCA(n_components=config.n_dim)
    result = model.fit_transform(X=emb)
    for i in range(config.n_dim):
        print(i + 1, sum(model.explained_variance_ratio_[:i + 1]))
    return result


# clustering methods
def clustering_kmeans(embeddings, n_clusters):
    if not os.path.exists(
            os.path.join(PICKLE_PATH, '{0}_{1}_cluster{2}'.format(config.dataset, config.method, config.n_clusters))):
        model = KMeans(n_clusters=n_clusters)
        model = model.fit(embeddings)
        pickle.dump(model, open(
            os.path.join(PICKLE_PATH, '{0}_{1}_cluster{2}'.format(config.dataset, config.method, config.n_clusters)),
            'wb'))
        return model.predict(embeddings)
    else:
        model = pickle.load(open(
            os.path.join(PICKLE_PATH, '{0}_{1}_cluster{2}'.format(config.dataset, config.method, config.n_clusters)),
            'rb'))
        return model.predict(embeddings)


def clustering_spectral(embeddings, n_clusters):
    print('Initializing spectralclustering model')
    model = SpectralClustering(n_clusters=n_clusters, eigen_solver='arpack', affinity='nearest_neighbors')
    print('Fitting spectralclustering model')
    model = model.fit(embeddings)
    print('Pred')
    if hasattr(model, 'labels_'):
        y_pred = model.labels_.astype(np.int)
    else:
        y_pred = model.predict(embeddings)
    return y_pred


# Save result as image
def make_plot(clustered_result):
    plt.figure()

    for cluster_idx, coord in clustered_result:
        plt.scatter(x=coord[0], y=coord[1], c=COLOR_MAP[cluster_idx], s=0.5)

    plt.savefig(os.path.join(BASE_DIR, 'figures', 'f_' + config.dimension_reduction + '_' + config.clustering))
    # pass


def summarize_clustering_result(pred, target_doc_idx=None):
    docs = pickle.load(open(os.path.join(PICKLE_PATH, 'wikipedia_docs'), 'rb'))
    labels = pickle.load(open(os.path.join(PICKLE_PATH, 'wikipedia_labels'), 'rb'))
    groups = [list() for _ in range(config.n_clusters)]

    if target_doc_idx is None:
        for idx, cluster_idx in enumerate(pred):
            groups[cluster_idx].append((idx, docs[idx]))
    else:
        for idx, cluster_idx in zip(target_doc_idx, pred):
            groups[cluster_idx].append((idx, docs[idx]))

    with open(os.path.join(BASE_DIR, 'output'), 'w', encoding='utf-8') as output_io:
        for idx, group in enumerate(groups):
            output_io.write('Cluster {0}\n'.format(idx))
            output_io.write('# docs: {0}\n'.format(len(group)))
            for keys, category in WIKIPEIDA_CATEGORIES:
                target_docs = [doc for idx, doc in group if category in labels[idx]]
                output_io.write(
                    '# {0}:\t{1}\t({2:.2f}%)\n'.format(category, len(target_docs), 100 * len(target_docs) / len(group)))
            for idx, doc in group:
                output_io.write('\t{0} {1}\n'.format(doc, labels[idx]))
            output_io.write('=======================\n')
            # painter_docs = [doc for index, doc in group if labels[index] == {'painter'}]


def main(config):
    "clustering"
    v = pickle.load(open(os.path.join(PICKLE_PATH, '{0}_v_thr{1}'.format(config.dataset, config.threshold)), 'rb'))

    if config.method == 'spectral_clustering':
        pred = clustering_spectral(embeddings=v, n_clusters=config.n_cluster)
    elif config.method == 'kmeans_clustering':
        pred = clustering_kmeans(embeddings=v, n_clusters=config.n_clusters)
    else:
        pred = None

    summarize_clustering_result(pred, target_doc_idx=target_doc_index)

    "classifier"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='wikipedia')
    parser.add_argument('--method', type=str, default='kmeans_clustering')
    parser.add_argument('--n_clusters', type=int, default=30)
    parser.add_argument('--threshold', type=str, default='0.001')
    config = parser.parse_args()
    print(config)
    main(config)
