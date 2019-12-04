def get_graph(dataset):
    if dataset == 'cora':
        from .undirected import CoraDataset
        return CoraDataset()
    else:
        raise ValueError
