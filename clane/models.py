import torch
from torch import nn


torch.manual_seed(0)

class BaseEdgeProbability(nn.Module):
    def __init__(self, dim):
        super(BaseEdgeProbability, self).__init__()
    
    def forward(self, z_src, z_dst):
        # z_src : B x       d
        # z_dst : B x |N| x d
        
        Asrc = self.A(z_src) # B x d = Bxd X dxd
        Bdst = self.B(z_dst) # B x |N| x d = Bx|N|xd X dxd
        return torch.matmul(Asrc.unsqueeze(1), Bdst.transpose(-1,-2)).sigmoid().squeeze(1)

    def get_sims(self, z_src, z_dst):
        
        Asrc = self.A(z_src)
        Bdst = self.B(z_dst)
        output = torch.matmul(Asrc.unsqueeze(1), Bdst.transpose(-1,-2))
        return output.squeeze(1)

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

