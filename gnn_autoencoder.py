import torch
import torch.nn.functional as F

class gnnlayer(torch.nn.Module):

    def __init__(self,input_shape,nodes):
        super().__init__()
        self.w = torch.nn.Parameter(
            torch.randn(input_shape,nodes)
            )

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

features = torch.tensor([[1.0, 0.0],   # Node 0
    [0.0, 1.0],   # Node 1
    [1.0, 1.0],   # Node 2
    [0.5, 0.5],   # Node 3
    [0.0, 0.0]
])   # Node 4

adj_matrix = compute_adj(edges)

class gnnencoder(torch.nn.Module):

    def __init__(self,input_dim):
        super().__init__()
        torch.manual_seed(42)
        self.layer1 = gnnlayer(2,32)
        self.layer2 = gnnlayer(32,64)
        self.layer3 = gnnlayer(64,32)

    def forward(self,edges,features):
        x = self.layer1(edges,features)
        x = self.layer2(edges, x)
        return self.layer3(edges, x)


class decoder(torch.nn.Module):

    def __init__(self, input_dim):
        super().__init__()
        self.layer1 = gnnlayer(input_dim,64)
        self.layer2 = gnnlayer(64,32)
        self.layer3 = gnnlayer(32,2)
    
    def forward(self,edges,features):
        x = self.layer1(edges, features)
        x = self.layer2(edges, x)
        return self.layer3(edges, x)


model_encoder  = gnnencoder(features.size(1))
result = model_encoder(adj_matrix,features)
model_decoder = decoder(result.size(1))
final_result = model_decoder(adj_matrix, result)

print(result)
print(final_result)
