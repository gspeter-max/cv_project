from sklearn.metrics import confusion_matrix, precision_recall_curve
import numpy as np
import polars
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import torch

torch.manual_seed(42)
np.random.seed(42)

df = polars.read_csv('/content/drive/MyDrive/synthetic_santander_realistic_chaos.csv')

df = df.with_columns(
    polars.col('ID_code').str.extract(r'_(\d+)').cast(polars.Int32)
)

# outliers
class_0 = df.filter(polars.col('target') == 0).height
class_1 = df.height - class_0

q1 = df['var_0'].quantile(0.25)
q3 = df['var_0'].quantile(0.75)
IQR = q3 - q1

lower_bound = q1 - (1.5 * IQR)
higher_bound = q3 + (1.5 * IQR)
df = df.filter( (df['var_0'] >= lower_bound) & (df['var_0'] <= higher_bound))
print(df)
'''
polars implimentation is more efficient compared to that sql in polars

context = polars.SQLContext()
context.register('df', df)
result = context.execute(f"""
            select *,
            from df
            where var_0 between {lower_bound} and {higher_bound}

    """).collect()

'''
x = df.drop('target','ID_code')
y = df['target']
x_train , x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2, random_state = 42, stratify = y)
x_train , x_val, y_train, y_val = train_test_split(x_train, y_train, test_size = 0.2 , random_state = 42,stratify= y_train)

sampler = SMOTE(random_state = 42)
x_train, y_train = sampler.fit_resample(x_train.to_numpy(), y_train.to_numpy())

x_train = torch.tensor(x_train, dtype = torch.float32)
x_test = x_test.to_torch(dtype = polars.Float32)
y_train = torch.tensor(y_train,dtype = torch.float32)
y_test = y_test.cast(polars.Float32).to_torch()
x_val = x_val.to_torch(dtype = polars.Float32)
y_val = y_val.cast(polars.Float32).to_torch()

class swise(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self,x):
        return x * torch.sigmoid(x)


class modelnn(torch.nn.Module):

    def __init__(self,input_shape):
        super().__init__()

        self.generator = torch.Generator().manual_seed(42)
        self.layer1 = torch.nn.Linear(input_shape , 256)
        self.layer1.is_special = False
        self.batch1 = torch.nn.GroupNorm(num_groups = 32,num_channels = 256)
        self.dropout1 = torch.nn.Dropout(0.2)
        self.swise = swise()

        self.layer2 = torch.nn.Linear(256 , 512)
        self.layer2.is_special = False
        self.batch2 = torch.nn.LayerNorm(512)
        self.dropout2 = torch.nn.Dropout(0.3)

        self.layer3 = torch.nn.Linear(512 , 256)
        self.layer3.is_special = False
        self.batch3 = torch.nn.LayerNorm(256)
        self.dropout3 = torch.nn.Dropout(0.3)

        self.layer4 = torch.nn.Linear(256 , 1)
        self.layer4.is_special = True

    def forward(self,input):

        x = self.swise(self.batch1(self.layer1(input)))
        x = self.dropout1(x)

        x = self.swise(self.batch2(self.layer2(x)))
        x = self.dropout2(x)

        x = self.swise(self.batch3(self.layer3(x)))
        x = self.dropout3(x)

        return torch.sigmoid(self.layer4(x))
g = torch.Generator().manual_seed(42)



def weight_initialing(w):
    if isinstance(w , torch.nn.Linear):
        fan_out, fan_in = w.weight.shape
        with torch.no_grad():
            if hasattr(w, 'is_special') and w.is_special:
                w.weight.data.copy_(
                    torch.randn(fan_out,fan_in,generator= g) * (1 / torch.sqrt(torch.tensor(fan_in)))
                )

            else :
                w.weight.data.copy_(
                    torch.randn(fan_out,fan_in,generator= g ) * (1.767 / torch.sqrt(torch.tensor(fan_in)))
                )
            if w.bias is None :
                w.bias.zeros_()

'''
# but that is too slow compared to above func

def weight_initialing(model):
    for name, Module in model.named_modules():
        if name != 'layer_4':
            Module.weight.data.copy_(torch.randn(fan_in,fan_out,generator= g ) * (1 / torch.sqrt(torch.tensor(fan_in))))
            if w.bias.data is None:
                w.bias.data.zeros_()
        else :
            Module.weight.data.copy_(torch.randn(fan_in,fan_out,generator= g ) * (1.767  / torch.sqrt(torch.tensor(fan_in))))
            if w.bias.data is None:
                w.bias.data.zeros_()
'''

class focal_loss(torch.nn.Module):

    def __init__(self,alpha = 0.53, gamma = 1.1):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        '''
        alpha ==> for class imblance handing
        gamma ( how much decrase the weight of majority class ) ===> for easy to classified examples
        '''

    def forward(self,y_pred, y_true):
        bce = torch.nn.functional.binary_cross_entropy(y_pred, y_true, reduction = 'none')

        pt = torch.where(y_true == 1 , y_pred, 1 - y_pred)
        pt = torch.clamp(pt, min = 1e-5 , max = 1 - 1e-5)

        focal_term = self.alpha * ( 1 - pt ) ** self.gamma
        '''first option '''
        # return focal_term * bce
        ''' second one '''
        result = focal_term * - torch.log(pt)
        return result


train_data = torch.utils.data.TensorDataset(x_train,y_train)
data_loader = torch.utils.data.DataLoader(train_data, batch_size = 256, shuffle = True)
model = modelnn(x_train.size(1))
model.layer4.is_special = True

model.apply(weight_initialing)

loss_func = focal_loss()
optimizer = torch.optim.Adam(model.parameters(), lr = 0.0001)
schuduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer ,
    'min',
    factor = 0.1,
    patience = 10

    )

pre_loss = 0 
patience = 0
min_loss_change = 0.00001 
for epoch in range(120):
    if patience >= 10 : 
        print('model is not improve more much ')
        break 

    model.train()

    for x_train, y_train in  data_loader:
        y_train = torch.unsqueeze(y_train, 1)
        optimizer.zero_grad()
        y_pred = model(x_train)
        loss = loss_func(y_pred, y_train).mean()
        loss.backward()
        optimizer.step()
        schuduler.step(loss)
        if torch.abs(loss - pre_loss) <= min_loss_change : 
            patience += 1 
        pre_loss = loss 

    if epoch % 10 == 0 :
        print(f'epoch --: {epoch}  loss ==>  {loss}')





model.eval()
with  torch.no_grad():
    test_y_pred_proba = model(x_test)
    val_y_pred_proba = model(x_val)


test_loss = loss_func(test_y_pred_proba, torch.unsqueeze(y_test, 1))
val_loss = loss_func(val_y_pred_proba, torch.unsqueeze(y_val, 1))

precision , recall, threshold = precision_recall_curve(y_test, test_y_pred_proba)

f2_score = 2 *( precision * recall) / ( precision + recall + 1e-8)
best_threshold = threshold[np.argmax(f2_score)]

test_y_pred =  (test_y_pred_proba > torch.tensor(best_threshold)).to(torch.int32)
val_y_pred =  (val_y_pred_proba > torch.tensor(best_threshold)).to(torch.int32)
print(f'confusion matrix : {confusion_matrix(torch.unsqueeze(y_test, 1), test_y_pred)}')
print(f'confusion matrix : {confusion_matrix(torch.unsqueeze(y_val, 1), val_y_pred)}')
