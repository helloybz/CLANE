def get_graph(dataset):
    if dataset == 'cora':
        from .undirected import CoraDataset
        return CoraDataset()
    if dataset == 'karate':
        from .undirected import KarateDataset
        return KarateDataset()
    else:
        raise ValueError
