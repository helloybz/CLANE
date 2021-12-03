import torch
import torch.nn as nn


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


class AsymmertricSimilarity(nn.Module):
    def __init__(
            self,
            n_dim: int,
    ) -> None:
        super(AsymmertricSimilarity, self).__init__()
        self.Phi_src = nn.Linear(n_dim, n_dim, bias=True)
        self.Phi_dst = nn.Linear(n_dim, n_dim, bias=True)

    def forward(
        self,
        z_src:  torch.Tensor,
        z_dst:  torch.Tensor,
    ) -> None:
        return self.Phi_src(z_src).dot(self.Phi_dst(z_dst))
