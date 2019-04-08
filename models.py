import torch
from torch import nn

from helper import normalize_elwise


class EdgeProbability(nn.Module):
    def __init__(self, dim):
        super(EdgeProbability, self).__init__()

        self.A = nn.Linear(in_features=dim,
                           out_features=dim,
                           bias=False)
        self.B = nn.Linear(in_features=dim,
                           out_features=dim,
                           bias=False)
        self.softmax = nn.Softmax(dim=0)

    def forward(self, z_srcs, zs_dests, **kwargs):
        # Z_srcs : (B x d)
        # Z_dests: (B x |N| x d)
        # Output : (B)
        if z_srcs.dim() == 1: z_srcs = z_srcs.unsqueeze(0)
        if zs_dests.dim() == 2: zs_dests = zs_dests.unsqueeze(0)
        AZ = self.A(z_srcs) # (B x d) * (d x d) = (B x d)
        AZ = torch.unsqueeze(input=AZ, dim=1)
        BZ_dests = self.B(zs_dests) # (B x |N| x d) * (d x d) = (B x |N| x d)
        BZ_dests = BZ_dests.transpose(1, 2) # (B x d x |N|)
        output = torch.matmul(AZ, BZ_dests) # (B x 1 x |N|)
        output = output.squeeze()
        if output.dim() == 0: output = output.unsqueeze(0)
        return torch.sigmoid(output) 
 
#    def get_similarities(self, z_srcs, zs_dests, **kwargs):
#        import pdb; pdb.set_trace()
#        if z_srcs.dim() == 1:
#            z_srcs = z_srcs.unsqueeze(0)
#        if zs_dests.dim() == 2:
#            zs_dests = zs_dests.unsqueeze(0)
#
#        return self.forward(z_srcs, zs_dests)
