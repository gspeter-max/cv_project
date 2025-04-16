from  pyspark.sql import SparkSession
spark = SparkSession.builder.appName('ctr').getOrCreate()
df = spark.read.csv('/content/drive/MyDrive/synthetic_ctr_data.csv',header = True, inferSchema = True)
df.repartition(8)
df.write.parquet("/content/sample_data/synthetic_ctr_data.parquet")

df_parquet = spark.read.parquet('/content/sample_data/synthetic_ctr_data.parquet')
import pyspark.sql.functions as f
df_parquet = df_parquet.dropDuplicates()

temp = {}
total_count = df_parquet.select(f.sum(f.lit(1))).collect()[0][0]

for col_ in df_parquet.columns:
    temp[col_] = total_count - df_parquet.agg(f.count(col_).alias('asfd')).collect()[0][0]


from pyspark.sql.types import *
def drop_fill(df,suggestion : str = 'drop'):
    if suggestion == 'drop':
        df = df.dropna()
        return df

    elif suggestion == 'fill':
        for field in df.schema :
            if isinstance(field.dataType, (IntegerType,DoubleType)):
                mean_ = df.agg(f.mean(field.name).alias('mean')).collect()[0][0]
                df = df.fillna(mean_,subset = field.name )
            elif isinstance(field.dataType,StringType):
                mode_ = df.agg(f.mode(field.name).alias('mode')).collect()[0][0]
                df = df.fillna(mode_,subset = field.name)
        return df
df_parquet = drop_fill(df_parquet,suggestion='fill')

df_parquet.columns
ploting_df = df_parquet.toPandas()
ploting_df = ploting_df.iloc[:100,:]
ploting_df = ploting_df.drop(columns = ['id','user_id','ad_campaign_id','site_id','site_domain'])
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

fig, axes = plt.subplots(5 , 2,figsize = (16, 9))
for i, col in enumerate(ploting_df.columns):
    if np.issubdtype(ploting_df[col].dtype,np.number):
        sns.distplot(ploting_df[col],ax= axes[int(np.floor(i/2)),i - 2* int(np.floor(i/2))])
    elif np.issubdtype(ploting_df[col].dtype,object):
        new_col = ploting_df[col].apply(lambda x: x[:5] + '...' if len(x) > 5  else x)
        sns.countplot(new_col,ax= axes[int(np.floor(i/2)),i -2* int(np.floor(i/2))])

plt.show()
df_parquet.columns
x = df_parquet.drop('id','user_id','site_id','click').toPandas()
y = df_parquet.select('click').toPandas()

from sklearn.model_selection import StratifiedShuffleSplit
sss = StratifiedShuffleSplit(n_splits = 1, test_size = 0.2 , random_state = 42)
train_idx , test_idx = next(sss.split(x,y))
x_train_ , x_test_ = x.loc[train_idx,:], x.loc[test_idx,:]
y_train_ , y_test_ = y.loc[train_idx,:],y.loc[test_idx,:]


from sklearn.preprocessing import TargetEncoder

encoder = TargetEncoder(categories='auto', target_type = 'binary')
columns_list = x.select_dtypes(('object','category')).columns
encoder.fit(x_train_[columns_list],y_train_)
x_train_[columns_list] = encoder.transform(x_train_[columns_list])
x_test_[columns_list] = encoder.transform(x_test_[columns_list])

from sklearn.preprocessing import QuantileTransformer

qt = QuantileTransformer(n_quantiles=900, output_distribution='normal')
x_train_ = qt.fit_transform(x_train_)
x_test_ = qt.transform(x_test_)

# torch model
import torch
import torch.nn.functional as F
from imblearn.over_sampling import SMOTE
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import TensorDataset, DataLoader

sampler = SMOTE(random_state = 42)
x_train_sampled ,y_train_sampled = sampler.fit_resample(x_train_, y_train_)

x_train = torch.tensor(x_train_sampled, dtype = torch.float32)
x_test = torch.tensor(x_test_, dtype = torch.float32)
y_train = torch.tensor(y_train_sampled.to_numpy(), dtype = torch.float32)
y_test = torch.tensor(y_test_.to_numpy(), dtype = torch.float32)


# torch model
import torch
import torch.nn.functional as F
from imblearn.over_sampling import SMOTE
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import TensorDataset, DataLoader

sampler = SMOTE(random_state = 42)
x_train_sampled ,y_train_sampled = sampler.fit_resample(x_train_, y_train_)

x_train = torch.tensor(x_train_sampled, dtype = torch.float32)
x_test = torch.tensor(x_test_, dtype = torch.float32)
y_train = torch.tensor(y_train_sampled.to_numpy(), dtype = torch.float32)
y_test = torch.tensor(y_test_.to_numpy(), dtype = torch.float32)

class ctr_model(torch.nn.Module): 
    
    def __init__(self,input_shape): 
        super().__init__()
        self.input_shape = input_shape

        self.sub_model1 =  torch.nn.Sequential(
            torch.nn.Linear(input_shape,64), 
            torch.nn.BatchNorm1d(64), 
            torch.nn.LeakyReLU(0.2), 
            torch.nn.Dropout(0.2)
        )

        self.sub_model2 = torch.nn.Sequential(
            torch.nn.Linear(64,128), 
            torch.nn.BatchNorm1d(128), 
            torch.nn.LeakyReLU(128), 
            torch.nn.Dropout(0.4)
        )

        self.residule = torch.nn.Linear(64,128)
        
        self.final_result = torch.nn.Sequential(
            torch.nn.Linear(128,64), 
            torch.nn.LayerNorm(64), 
            torch.nn.LeakyReLU(64), 
            torch.nn.Linear(64,1)
        )

    def forward(self, x): 
        x1 = self.sub_model1(x)
        x2 = self.sub_model2(x1)
        x2 =+ self.residule(x1)
        c = self.final_result(x2)
        return torch.sigmoid(c)
    


def custom_loss(y_true, y_pred):
    # Class-aware focal parameters
    pos_weight = y_true.shape[0] / (2 * (y_true.sum() + 1))  # Dynamic weighting
    neg_weight = y_true.shape[0] / (2 * ((1 - y_true).sum() + 1))
    
    # Asymmetric focal loss
    bce = F.binary_cross_entropy(y_pred, y_true, reduction='none')
    pt = torch.exp(-bce)
    focal_loss = (pos_weight * (1 - pt[y_true == 1]) ** 2.5 * bce[y_true == 1]).mean() + \
                 (neg_weight * (1 - pt[y_true == 0]) ** 1.5 * bce[y_true == 0]).mean()

    # Gradient-stable F-beta calculation
    tp = (y_true * y_pred).sum()
    fp = ((1 - y_true) * y_pred).sum()
    fn = (y_true * (1 - y_pred)).sum()
    
    precision = tp / (tp + fp + 1e-10)
    recall = tp / (tp + fn + 1e-10)
    
    f2 = (5 * precision * recall) / (4 * precision + recall + 1e-10)
    
    # Adaptive loss balancing
    return 0.6 * focal_loss + 0.4 * (1 - f2) + 0.1 * (fp/(y_true.shape[0] + 1e-7))
    


from torch.utils.data import WeightedRandomSampler

class_weights = 1. / torch.bincount(y_train.int().squeeze())
samples_weights = class_weights[y_train.int().squeeze()]
sampler = WeightedRandomSampler(samples_weights, len(samples_weights), replacement=True)

tensor_data = TensorDataset(x_train, y_train)
data_loader = DataLoader(tensor_data, batch_size=64, sampler=sampler)
# Hyperparameter configuration
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-3)
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, 
                                             max_lr=5e-4,
                                             steps_per_epoch=len(data_loader),
                                             epochs=100)

# DataLoader with aggressive oversampling

from torcheval.metrics.functional import binary_f1_score

for epoch in range(500):
    model.train()
    total_loss = 0
    for x_train, y_train in data_loader:

        optimizer.zero_grad()
        y_pred = model(x_train)
        loss = custom_loss(y_train, y_pred)
        loss.backward()
        optimizer.step()
        scheduler.step(loss)
        total_loss += loss
    if epoch % 20 == 0 :
        print(f'epoch : {epoch} | loss : {total_loss / len(data_loader)}')

from sklearn.metrics import confusion_matrix, recall_score , precision_score

y_pred_ = model(x_test)
model.eval()
with torch.no_grad():
    y_pred_ = model(x_test)


y_pred_numpy = y_pred_.detach().numpy()
y_test_numpy = y_test.detach().numpy()
y_pred_numpy = (y_pred_numpy > 0.5).astype(int)
recall = recall_score(y_test_numpy,y_pred_numpy)
precision  = precision_score(y_test_numpy,y_pred_numpy)
confusion = confusion_matrix(y_test_numpy,y_pred_numpy)
print(f'recall  : {recall}')
print(f'precision  : {precision}')
print(f'confusion  : {confusion}')
