from clane import g
from .. import similarities


def get_similarity(feature_dim=None):
    _measure = g.config.similarity
    if not isinstance(_measure, str):
        raise ValueError

    if _measure.upper() == 'COSINE':
        return similarities.CosineSimilarity()
    elif _measure.upper() == 'ASS':
        return similarities.AsymmetricSingleScalar(dim=feature_dim)
    elif _measure.upper() == 'AMS':
        return similarities.AsymmetricMultiScalar(dim=feature_dim)
    else:
        raise ValueError
