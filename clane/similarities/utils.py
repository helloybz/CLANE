def get_similarity(measure, **kwargs):
    if measure == 'cosine':
        from .nonparametric import CosineSimilarity
        return CosineSimilarity()

    elif measure == 'ASS':
        from .parametric import AsymmetricSingleScalar
        return AsymmetricSingleScalar(dim=kwargs['dim'])

    else:
        raise ValueError
