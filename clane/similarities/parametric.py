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
        self.A = nn.Linear(dim, dim, bias=False,)
        self.B = nn.Linear(dim, dim, bias=False)

    def forward(self, z_src, z_dst):
        return self.A(z_src).unsqueeze(-2).matmul(
            self.B(z_dst).unsqueeze(-2).transpose(-2,-1)
        ).reshape([-1])


class AsymmetricMultiScalar(nn.Module, Parametric):
    def __init__(self, dim):
        super(AsymmetricMultiScalar, self).__init__()

        self.A = nn.Sequential(
            nn.Linear(dim, dim, bias=False,),
            nn.Sigmoid(),
            nn.Linear(dim, dim, bias=False,),
        )
        self.B = nn.Sequential(
            nn.Linear(dim, dim, bias=False,),
            nn.Sigmoid(),
            nn.Linear(dim, dim, bias=False,),
        )

    def forward(self, z_src, z_dst):
        return self.A(z_src).unsqueeze(-2).matmul(
            self.B(z_dst).unsqueeze(-2).transpose(-2,-1)
        ).reshape([-1])
