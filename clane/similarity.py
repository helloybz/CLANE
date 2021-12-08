import torch
import torch.nn as nn


class CosineSimilarity:
    """
    Cosine similarity between the two vector.

    Given two vector v1 and v2, the cosine similarity between the two vector
    is the cosine of theta, where the theta is the angle between the two vector on therir inner product space.

    The cosine of the theta can be derived from Euclidean dot product of the two vectors.
    """

    def __init__(
        self,
        **kwargs
    ) -> None:
        super(CosineSimilarity, self).__init__()

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
            **kwargs,
    ) -> None:
        super(AsymmertricSimilarity, self).__init__()
        self.Phi_src = nn.Linear(n_dim, n_dim, bias=False)
        self.Phi_dst = nn.Linear(n_dim, n_dim, bias=False)
        nn.init.xavier_normal_(self.Phi_src.weight)
        nn.init.xavier_normal_(self.Phi_dst.weight)

    def forward(
        self,
        z_src:  torch.Tensor,
        z_dst:  torch.Tensor,
    ) -> torch.Tensor:
        return self.Phi_src(z_src).unsqueeze(-2).matmul(self.Phi_dst(z_dst).unsqueeze(-1)).squeeze()
