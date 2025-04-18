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

def compute_adj(edges):
    num_nodes = torch.max(edges) + 1 
    adj_matrix = torch.zeros(
        num_nodes,num_nodes,
        dtype = torch.int32)
    
    for edge in edges: 
        u , v = edge[0].item() , edge[1].item()
        adj_matrix[u, u] = 1 
        adj_matrix[u, v] = 1 
    return adj_matrix

edges = torch.tensor([
    (0, 1),
    (0, 2),
    (1, 2),
    (2, 3),
    (3, 4),
    (4, 0)
])

features = [[1.0, 0.0],   # Node 0  
    [0.0, 1.0],   # Node 1  
    [1.0, 1.0],   # Node 2  
    [0.5, 0.5],   # Node 3  
    [0.0, 0.0]
]   # Node 4

adj_matrix = compute_adj(edges)

class gnnencoder(torch.nn.Module):

        def __init__(self,input_dim):
            self.layer1 = gnnlayer(input_dim,64)
            self.layer2 = gnnlayer(64,32)

        def forward(self, input):
            x = self.layer1(input)
            return self.layer2(x)

