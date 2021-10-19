import torch


class CosineSimilarity:
    """
    Cosine similarity between the two vector.

    Given two vector v1 and v2, the cosine similarity between the two vector
    is the cosine of theta, where the theta is the angle between the two vector on therir inner product space.

    The cosine of the theta can be derived from Euclidean dot product of the two vectors.
    """

    def __call__(
        self,
        v1: torch.Tensor,
        v2: torch.Tensor,
    ) -> torch.Tensor:
        return v1.dot(v2).div(v1.pow(2).sum().sqrt() * v2.pow(2).sum().sqrt())
