import torch.nn as nn


class NonParametric:
    @property
    def is_nonparametric(self):
        return True


class CosineSimilarity(nn.Module, NonParametric):
    def __init__(self):
        super(CosineSimilarity, self).__init__()

    def forward(self, z_src, z_dst):
        return nn.CosineSimilarity(dim=-1)(z_src, z_dst)
