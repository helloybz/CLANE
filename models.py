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

    def get_sims(self, z, U):
        Az = self.A(z) # (d) * (d x d) = (d)
        BU = self.B(U) # (|N| x d) * (d x d) = (|N| x d)
        output = BU.mv(Az) # (|N| x d) = (|N|)
         
        return output


if __name__ == "__main__":
    from dataset import CoraDataset
    import torch
    from torch.utils.data import DataLoader
    import pdb

    graph = CoraDataset(device=torch.device('cuda'))

    model = EdgeProbability(dim=graph.Z.shape[1]).to(torch.device('cuda'))

    model(graph.Z[(graph.A==1).nonzero()])



