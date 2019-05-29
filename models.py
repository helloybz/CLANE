import torch
from torch import nn

class EdgeProbability(nn.Module):
    def __init__(self, dim):
        super(EdgeProbability, self).__init__()

        self.A = nn.Linear(in_features=dim,
                           out_features=dim,
                           bias=False)
        self.B = nn.Linear(in_features=dim,
                           out_features=dim,
                           bias=False)
    
    def forward(self, z_src, z_dst):
        Asrc = self.A(z_src).unsqueeze(-2)
        Bdst = self.B(z_dst).unsqueeze(-1)
        return torch.matmul(Asrc,Bdst).sigmoid().view(-1)

    def get_sims(self, z_src, z_dst):
        Asrc = self.A(z_src).unsqueeze(-2)
        Bdst = self.B(z_dst).unsqueeze(-1)
        output = torch.matmul(Asrc, Bdst)
        return output.view(-1)


if __name__ == "__main__":
    from dataset import CoraDataset
    import torch
    from torch.utils.data import DataLoader
    import pdb

    graph = CoraDataset(device=torch.device('cuda'))

    model = EdgeProbability(dim=graph.Z.shape[1]).to(torch.device('cuda'))

    model(graph.Z[(graph.A==1).nonzero()])



