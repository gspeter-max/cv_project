import torch
import torch.nn.functional as F

class gnnlayer(torch.nn.Module):

def __init__(self,input_shape,nodes):  
    super().__init__()  
    self.w = torch.nn.Parameter(torch.randn(input_shape,nodes))  

def forward(self,adj,features):  
    i = torch.eye(adj.size(0),device = adj.device)  
    a_hat = adj + i  
                                                                           sums = torch.sum(a_hat,dim = 1)  
    norm_term = torch.pow(sums,-0.5)  
    norm_term_adj = torch.diag(norm_term)  
                                                                           normalize = norm_term_adj @ a_hat @ norm_term_adj  
    n_f = torch.matmul(normalize,features)  

    z = torch.matmul(n_f ,self.w)  

    return F.relu(z)

