import torch

x = torch.tensor(df_balanced.drop('target',axis = 1).to_numpy(),
dtype = torch.float32
)
y = torch.tensor(df_balanced['target'].to_numpy(),
dtype = torch.float32
)

import torch
import matplotlib.pyplot as plt
import seaborn as sns
'''
f  = torch.tensor([
[1.0, 0.0],
[0.9, 0.1],
[0.0, 1.0],
[0.1, 0.9],
[1.0, 0.1],
[0.2, 0.2],
[0.0, 0.9],
[0.8, 0.2],
[0.5, 0.5],
[1.0, 1.0]
]*100, dtype = torch.float32 )
'''

features = x / torch.sum(x,  dim = 0)
con_matrix = features @ features.T
_90_quantile = torch.quantile(torch.flatten(
con_matrix
),q = 0.90)
adj_matrix = (con_matrix > _90_quantile).to(
torch.int32
)
i = torch.eye(adj_matrix.size(0),dtype = torch.float32)
adj_matrix = torch.clip(adj_matrix + i, 0,1)

that is for gnn

'''
y_true = torch.tensor([1, 1, 0, 0, 1, 0, 0, 1, 0, 1]100, dtype=torch.float32y_true = y_true.view(10100,1)
y_true.shape
'''
y = y.view(1000,1)
import torch

class mlp(torch.nn.Module):

def __init__(self, input_shape):  
    super().__init__()  
    self.layer1 = torch.nn.Linear(input_shape,32)  
    self. layer2 = torch.nn.Linear(32,64)  
    self. layer3 = torch.nn.Linear(64,32)  
    self.layer4 = torch.nn.Linear(32,1)  

def forward(self,adj_matrix):  
    x = torch.nn.functional.relu(self.layer1(adj_matrix))  
    x = torch.nn.functional.relu(self.layer2(x))  
    x = torch.nn.functional.relu(self.layer3(x))  
    return torch.nn.functional.sigmoid(self.layer4(x))

def loss_func(y_true,y_pred):
bce = torch.nn.functional.binary_cross_entropy(
y_pred,y_true,weight = torch.full_like(
y_true,9.0),reduction = 'none')

N = len(y_true)  

p_w = N / (2 * torch.sum(y_true))  
n_w = N / (2 * torch.sum(1 - y_true))  

focal = (p_w * ((1 - y_pred[y_true == 1])**2.5)*bce[y_true == 1]).mean() + (n_w * ((1 - y_pred[y_true == 0])**1.5)*bce[y_true == 0]).mean()  
return focal

model = mlp(adj_matrix.size(0))
result = model(adj_matrix)

from torch.optim import Adam
optimizer = Adam(model.parameters(), lr = 0.001 )
for epoch in range(100):
optimizer.zero_grad()
y_pred = model(adj_matrix)
loss = loss_func(y,y_pred)
loss.backward()
optimizer.step()

if epoch % 10 == 0:  
    print(f' epoch --: {epoch} loss - {loss}')
