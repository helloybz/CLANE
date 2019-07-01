import torch
from torch import nn

torch.manual_seed(0)
class EdgeProbability(nn.Module):
    def __init__(self, dim):
        super(EdgeProbability, self).__init__()

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
#        self.A = nn.Linear(in_features=dim,
#                           out_features=dim,
#                           bias=False)
#        self.B = nn.Linear(in_features=dim,
#                           out_features=dim,
#                           bias=False)
    
    def forward(self, z_src, z_dst):
        Asrc = self.A(z_src)
        Bdst = self.B(z_dst)
        return torch.matmul(Asrc, Bdst.t()).sigmoid().view(-1)


    def get_sims(self, z_src, z_dst):
        Asrc = self.A(z_src)
        Bdst = self.B(z_dst)
        output = torch.matmul(Asrc, Bdst.t())
        return output.view(-1)


if __name__ == "__main__":
    from dataset import CoraDataset
    import torch
    from torch.utils.data import DataLoader
    import pdb

    graph = CoraDataset(device=torch.device('cuda'))

    model = EdgeProbability(dim=graph.Z.shape[1]).to(torch.device('cuda'))

    model(graph.Z[(graph.A==1).nonzero()])



