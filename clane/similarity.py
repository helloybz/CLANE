import torch
import torch.nn as nn


class Similarity:
    def is_trainable(self):
        return isinstance(self, nn.Module)


class CosineSimilarity(Similarity):
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
        if v1.dim() == 1:
            v1 = v1.unsqueeze(0)
        if v2.dim() == 1:
            v2 = v2.unsqueeze(0)
        v1 = v1.unsqueeze(1)
        v2 = v2.unsqueeze(-1)
        return v1.matmul(v2).squeeze(1).squeeze(1).div(v1.pow(2).sum().sqrt() * v2.pow(2).sum().sqrt())


class AsymmertricSimilarity(nn.Module, Similarity):
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
