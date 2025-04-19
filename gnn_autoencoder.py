import torch
import torch.nn.functional as F

class gnnlayer(torch.nn.Module):

    def __init__(self,input_shape,nodes):
        super().__init__()
        self.w = torch.nn.Parameter(
            torch.randn(input_shape,nodes)
            )

    def forward(self,adj,features, function  = None):
        i = torch.eye(adj.size(0),device = adj.device)
        a_hat = adj + i
        sums = torch.sum(a_hat,dim = 1)
        norm_term = torch.pow(sums,-0.5)

        norm_term_adj = torch.diag(norm_term)
        normalize = norm_term_adj @ a_hat @ norm_term_adj
        n_f = torch.matmul(normalize,features)
        z = torch.matmul(n_f ,self.w)
        if function == 'relu':
            return F.relu(z)
        elif function == 'sigmoid':
            return F.sigmoid(z)

        return z

def compute_adj(edges):
    num_nodes = torch.max(edges) + 1
    adj_matrix = torch.zeros(
        num_nodes,num_nodes,
        dtype = torch.int32)

    for edge in edges:
        u , v = edge[0].item() , edge[1].item()
        adj_matrix[u, v] = 1
    return adj_matrix
'''
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
print(adj_matrix)
'''

class gnnencoder(torch.nn.Module):

    def __init__(self,input_dim):
        super().__init__()
        torch.manual_seed(42)
        self.layer1 = gnnlayer(input_dim,32)
        self.batch1 = torch.nn.BatchNorm1d(32)
        self.layer2 = gnnlayer(32,64)
        self.batch2 = torch.nn.BatchNorm1d(64)
        self.layer3 = gnnlayer(64,32)

    def forward(self,edges,features):
        x = self.batch1(self.layer1(edges,features,
                                    function = 'relu'))
        x = self.batch2(self.layer2(edges, x,
                                    function = 'relu'))
        return self.layer3(edges, x)

class decoder(torch.nn.Module):

        def __init__(self, input_dim):
            super().__init__()
            self.layer1 = gnnlayer(input_dim,64)
            self.batch1 = torch.nn.BatchNorm1d(64)
            self.layer2 = gnnlayer(64,32)
            self.batch2 = torch.nn.BatchNorm1d(32)
            self.layer3 = gnnlayer(32,2)

        def forward(self,edges,features):
            x = self.layer1(edges, features,
                    function = 'relu'
                )
            x = self.batch1(x)
            x = self.layer2(edges , x,
                function = 'relu'
                    )
            x = self.batch2(x)
            return self.layer3(edges, x)

'''
from torch.nn import BCELoss

model_encoder  = gnnencoder(features.size(1))
result = model_encoder(adj_matrix,features)
model_decoder = decoder(result.size(1))
final_result = model_decoder(adj_matrix, result)
similarity = final_result @ final_result.T
y_pred_prob  = torch.nn.functional.sigmoid(similarity)
#y_pred = torch.flatten((y_pred_prob > 0.4).to(torch.float32))
y_true = adj_matrix.to(torch.float32)
loss = BCELoss()
loss = loss(y_true,y_pred_prob)
'''

import networkx as nx
import numpy as np

# Create a random graph with 1000 nodes and around 3000 edges
num_nodes = 1000
num_edges = 3000

# Generate a random graph
G = nx.gnm_random_graph(num_nodes, num_edges)
# Generate a feature matrix (1000 nodes, 16 features per node)
feature_dim = 16
features = torch.rand((num_nodes, feature_dim), dtype=torch.float32)

# Create adjacency matrix
adj_matrix = nx.to_numpy_array(G)
adj_tensor = torch.tensor(adj_matrix, dtype=torch.float32)


class staked_model(torch.nn.Module):

    def __init__(self, input_shape):
        super().__init__()
        self.encoder = gnnencoder(input_shape)
        self. decoder = decoder(32)

    def forward(self,adj_tensor, features):
        x = self.encoder(adj_tensor, features )
        x = self.decoder(adj_tensor, x)
        x = x @ x.T
        x = torch.nn.functional.sigmoid(x)
        return x

model = staked_model(features.size(1))
result = model(adj_tensor,features)

from torch.optim import Adam
from torch.nn import BCELoss
from torch.optim.lr_schuduler import ReduceLROnPlateau

optimizer  = Adam(model.parameters(), lr = 0.001)
loss_func = BCELoss()
scheduler = ReduceLROnPlateau(optimizer,mode = 'min', patience= 5)

for epoch in range(100):
    model.train()
    model.zero_grad()


    y_pred = model(adj_tensor,features).to(torch.float32)
    adj = adj_tensor.to(torch.float32)
    loss = loss_func(y_pred,adj)
    loss.backward()
    optimizer.step()
    scheduler.step(loss) 

    if epoch % 10 == 0:
        print(f'epochs --: {epoch}  loss - {loss}')

prediction = model(adj_tensor,features)
print((prediction > 0.5).to(torch.int32))
print(adj_tensor)