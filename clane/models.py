import torch
from torch import nn


torch.manual_seed(0)

class BaseEdgeProbability(nn.Module):
    def __init__(self, dim):
        super(BaseEdgeProbability, self).__init__()
    
    def forward(self, z_src, z_dst):
        Asrc = self.A(z_src)
        Bdst = self.B(z_dst)
        return torch.matmul(Asrc, Bdst.t()).sigmoid().view(-1)


    def get_sims(self, z_src, z_dst):
        Asrc = self.A(z_src)
        Bdst = self.B(z_dst)
        output = torch.matmul(Asrc, Bdst.t())
        return output.view(-1)


class MultiLayer(BaseEdgeProbability):
    def __init__(self, dim):
        super(MultiLayer, self).__init__(dim)

        self.A = nn.Sequential(
            nn.Linear(dim,dim),
            nn.Tanh(),
            nn.Linear(dim,dim)
        )
        self.B = nn.Sequential(
            nn.Linear(dim,dim),
            nn.Tanh(),
            nn.Linear(dim,dim)
        )


class SingleLayer(BaseEdgeProbability):
    def __init__(self, dim):
        super(SingleLayer, self).__init__(dim)

        self.A = nn.Linear(dim, dim)
        self.B = nn.Linear(dim, dim)

