import torch.nn as nn


class Parametric:
    @property
    def is_nonparametric(self):
        return False


class AsymmetricSingleScalar(nn.Module, Parametric):
    def __init__(self, dim):
        super(AsymmetricSingleScalar, self).__init__()
        # TODO: Consider set bias as False
        self.A = nn.Linear(dim, dim)
        self.B = nn.Linear(dim, dim)

    def forward(self, v_src, v_dst):
        return self.A(v_src).matmul(self.B(v_dst).transpose(1, 2)).squeeze()
