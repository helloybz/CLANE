"""
Evaluate

1. Clustering
"""
import argparse
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer

from settings import PICKLE_PATH, BASE_DIR, WIKIPEDIA_CATEGORIES, ART_MOVEMENTS

# local helpers
COLOR_MAP = ['red', 'blue', 'green']


def _load_embedding(pickle_name):
    return pickle.load(open(os.path.join(PICKLE_PATH, pickle_name), 'rb'))


# dimension reductions
def dim_reduce_pca(emb):
    model = PCA(n_components=config.n_dim)
    result = model.fit_transform(X=emb)
    for i in range(config.n_dim):
        print(i + 1, sum(model.explained_variance_ratio_[:i + 1]))
    return result


# clustering methods
def clustering_kmeans(embeddings, n_clusters):
    model = KMeans(n_clusters=n_clusters)
    model = model.fit(embeddings)
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


def classifying_svc(embeddings, labels):
    clf = svm.SVC(kernel='linear')
    pass


# Save result as image
def make_plot(clustered_result):
    plt.figure()

    for cluster_idx, coord in clustered_result:
        plt.scatter(x=coord[0], y=coord[1], c=COLOR_MAP[cluster_idx], s=0.5)

    plt.savefig(os.path.join(BASE_DIR, 'figures', 'f_' + config.dimension_reduction + '_' + config.clustering))
    # pass


def summarize_clustering_result(pred, v, target_doc_idx=None):
    docs = pickle.load(open(os.path.join(PICKLE_PATH, 'wikipedia_docs'), 'rb'))
    labels = pickle.load(open(os.path.join(PICKLE_PATH, 'wikipedia_labels'), 'rb'))
    groups = [list() for _ in range(config.n_clusters)]

    if target_doc_idx is not None:
        # docs = [doc for idx, doc in enumerate(docs) if idx in target_doc_idx]
        # labels = [label for idx, label in enumerate(labels) if idx in target_doc_idx]

        for doc_idx, cluster_idx in zip(target_doc_idx, pred):
            groups[cluster_idx].append((doc_idx, docs[doc_idx]))

    else:
        for doc_idx, cluster_idx in enumerate(pred):
            groups[cluster_idx].append((doc_idx, docs[doc_idx]))

    if config.score_metric == 'silhouette':
        average_score = silhouette_score(X=v, labels=pred, sample_size=int(len(pred) * 0.75))
    elif config.score_metric == 'none':
        pass
    else:
        average_score = silhouette_score(X=v, labels=pred)

    with open(os.path.join(BASE_DIR, 'output'), 'w', encoding='utf-8') as output_io:
        if config.score_metric != 'none':
            output_io.write('Average {} Score: {}\n'.format(config.score_metric, average_score))
        else:
            pass

        for group_idx, group in enumerate(groups):
            output_io.write('Cluster {0}\n'.format(group_idx))
            output_io.write('\tNumber of docs: {0}\n'.format(len(group)))
            for keys, category in WIKIPEDIA_CATEGORIES:
                target_docs = [doc for doc_idx, doc in group if category in labels[doc_idx]]
                if len(target_docs) != 0:
                    try:
                        output_io.write(
                            '# {0}:\t{1}\t({2:.2f}%)\n'.format(category, len(target_docs),
                                                               100 * len(target_docs) / len(group)))
                    except ZeroDivisionError:
                        pass

            for idx, doc in group:
                output_io.write('\t{0} {1}\n'.format(doc, labels[idx]))
            output_io.write('=======================\n')
            # painter_docs = [doc for index, doc in group if labels[index] == {'painter'}]


def summarize_classifying_result(pred):
    print(pred)


def main(config):
    labels = pickle.load(open(os.path.join(PICKLE_PATH, 'wikipedia_labels'), 'rb'))

    v = pickle.load(open(os.path.join(PICKLE_PATH, '{}_v_{}_txt{}_thr{}_a{}'.format(
        config.dataset,
        config.img_embedder,
        config.doc2vec_size,
        config.threshold,
        config.alpha,
        config.sim_metric,
    )), 'rb'))

    if config.target_class != 'all':
        target_doc_idx = [idx for idx, label in enumerate(labels) if config.target_class in label]
        v = v[target_doc_idx,]
        labels = [label for idx, label in enumerate(labels) if config.target_class in label]
        del target_doc_idx

    # split into train & test if classifying.
    if 'classifying' in config.method:
        mlb = MultiLabelBinarizer()
        labels = mlb.fit_transform(labels)
        v_train, v_test, y_train, y_test = train_test_split(v, labels)
        del mlb

    if config.method == 'spectral_clustering':
        pred = clustering_spectral(embeddings=v, n_clusters=config.n_cluster)
    elif config.method == 'kmeans_clustering':
        pred = clustering_kmeans(embeddings=v, n_clusters=config.n_clusters)
    elif config.method == 'SVC_classifying':
        clf = svm.SVC(kernel='linear')
        multi_label_clf = OneVsRestClassifier(clf)
        multi_label_clf.fit(X=v_train, y=y_train)
        multi_label_clf.predict(X=v_test)
        pred = multi_label_clf.score(X=v_test, y=y_test)
    else:
        raise ValueError

    if 'clustering' in config.method:
        summarize_clustering_result(pred=pred, target_doc_idx=target_doc_idx, v=v)
    elif 'classifying' in config.method:
        summarize_classifying_result(pred=pred)
    else:
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # V config
    parser.add_argument('--dataset', type=str, default='wikipedia')
    parser.add_argument('--img_embedder', type=str, default='resnet152')
    parser.add_argument('--doc2vec_size', type=int, default=1024)
    parser.add_argument('--sim_metric', type=str, default='cosine_C')
    parser.add_argument('--alpha', type=float, default=0.9)
    parser.add_argument('--threshold', type=str, default='0.001')

    # Evaluation config
    parser.add_argument('--n_clusters', type=int, default=30)
    parser.add_argument('--score_metric', type=str, default='silhouette')
    parser.add_argument('--target_class', type=str, default='painter')
    parser.add_argument('--method', type=str, default='SVC_classifying')

    config = parser.parse_args()
    print(config)
    main(config)
