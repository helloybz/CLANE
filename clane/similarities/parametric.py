import torch.nn as nn


class Parametric(object):
    @property
    def is_nonparametric(self):
        return False

    def __name__(self):
        return self.__name__


class AsymmetricSingleScalar(nn.Module, Parametric):
    def __init__(self, dim):
        super(AsymmetricSingleScalar, self).__init__()
        # TODO: Consider set bias as False
        self.A = nn.Linear(dim, dim)
        self.B = nn.Linear(dim, dim)

    def forward(self, z_src, z_dst):
        return self.A(z_src).matmul(self.B(z_dst).transpose(1, 2)).reshape(-1)
