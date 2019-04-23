import torch
from torch import nn

class EdgeProbability(nn.Module):
    def __init__(self, dim):
        super(EdgeProbability, self).__init__()

        self.A = nn.Linear(in_features=dim,
                           out_features=dim,
                           bias=True)
        self.B = nn.Linear(in_features=dim,
                           out_features=dim,
                           bias=True)
    
        self.softmax = nn.Softmax(dim=0)
        self.relu = nn.ReLU()
        self.softsign = nn.Softsign()
 
    def forward(self, z, U, **kwargs):
        Az = self.A(z) # (d) * (d x d) = (d)
        BU = self.B(U) # (|N| x d) * (d x d) = (|N| x d)
        output = BU.mv(Az) # (|N| x d) = (|N|)
        output = output.sigmoid()
        return output
    
    def get_sims(self, z, U):
        Az = self.A(z) # (d) * (d x d) = (d)
        BU = self.B(U) # (|N| x d) * (d x d) = (|N| x d)
        output = BU.mv(Az) # (|N| x d) = (|N|)
         
        return output

