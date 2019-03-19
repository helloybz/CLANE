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

    def forward1(self, embedding_pairs):
        srcs  = embedding_pairs[:,0,:].clone()
        dests = embedding_pairs[:,1,:].clone()

        az = self.A(srcs)
        az = az.unsqueeze(-2)
        
        bz = self.B(dests)
        bz = bz.unsqueeze(-1)
        
        output = torch.matmul(az, bz)
        output = normalize_elwise(output)[0]
        output = torch.sigmoid(output)
        return output

    def forward(self, Z_srcs, Z_dests):
        # Z_srcs : (B x d)
        # Z_dests: (B x |N| x d)
        # Output : (B)
        AZ = self.A(Z_srcs) # (B x d) * (d x d) = (B x d)
        AZ = torch.unsqueeze(input=AZ, dim=1)
        BZ_dests = self.B(Z_dests) # (B x |N| x d) * (d x d) = (B x |N| x d)
        BZ_dests = BZ_dests.transpose(1, 2) # (B x d x |N|)
        output = torch.matmul(AZ, BZ_dests) # (B x 1 x 1)
        output.squeeze_() # (B)
        if output.dim() == 0: output.unsqueeze_(0)
        return torch.sigmoid(output) 
 
    def get_similarities(self, Z_srcs, Z_dests):
        probs = self.forward(Z_srcs.unsqueeze(0), Z_dests)
        softmax_probs = self.softmax(probs)
        return self.softmax(probs)

