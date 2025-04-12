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
x= df_parquet.drop('id','user_id','site_id','click').toPandas()
y = df_parquet.select('click').toPandas()

from sklearn.preprocessing import TargetEncoder
encoder = TargetEncoder(categories='auto', target_type = 'binary')
columns_list = x.select_dtypes(('object','category')).columns
encoder.fit(x[columns_list],y)
x[columns_list] = encoder.transform(x[columns_list])


from sklearn.model_selection import StratifiedShuffleSplit
sss = StratifiedShuffleSplit(n_splits = 1, test_size = 0.2 , random_state = 42)
train_idx , test_idx = next(sss.split(x,y))
x_train_ , x_test_ = x.loc[train_idx,:], x.loc[test_idx,:]
y_train_ , y_test_ = y.loc[train_idx,:],y.loc[test_idx,:]



# torch model
import torch
import torch.nn.functional as F

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train_)
x_test= scaler.transform(x_test_)
print(type(x_test))
x_train = torch.tensor(x_train, dtype = torch.float32)
x_test = torch.tensor(x_test, dtype = torch.float32)
y_train = torch.tensor(y_train_.to_numpy(), dtype = torch.float32)
y_test = torch.tensor(y_test_.to_numpy(), dtype = torch.float32)


class roc_model(torch.nn.Module):
    def __init__(self,input_dims):
        super().__init__()
        self.layer1 = torch.nn.Linear(input_dims,68)
        self.batch1 = torch.nn.BatchNorm1d(68)
        self.layer2 = torch.nn.Linear(68,32)
        self.batch2 = torch.nn.BatchNorm1d(32)
        self.layer3  = torch.nn.Linear(32,1)

    def forward(self,inputs):
        x = torch.relu(self.batch1(self.layer1(inputs)))
        x = torch.relu(self.batch2(self.layer2(x)))
        output = self.layer3(x)
        return output

def custom_loss(y_true, y_pred):
    pos_list = y_pred[y_true == 1]
    neg_list = y_pred[y_true == 0]

    if len(pos_list) == 0 or len(neg_list) == 0:
        return torch.tensor(0.0, dtype = y_pred.dtype)

    pos_list = pos_list.unsqueeze(1)
    neg_list = neg_list.unsqueeze(0)
    diff = pos_list - neg_list
    loss =  F.softplus(-diff)
    return torch.mean(loss)

model = roc_model(x_train.shape[1])
optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)

for epoch in range(100):
    optimizer.zero_grad()
    y_pred = model(x_train)
    loss = custom_loss(y_train, y_pred)
    loss.backward()
    optimizer.step()

    if epoch % 20 == 0:
        print(f'epoch - {epoch}   loss -- {loss}')


from sklearn.metrics import roc_auc_score

y_pred_ = model(x_test)
model.eval()
with torch.no_grad():
    y_pred_ = model(x_test)

y_pred_numpy = y_pred_.detach().numpy()
y_test_numpy = y_test.detach().numpy()
roc = roc_auc_score(y_test.numpy(),y_pred_.numpy())
print(f'roc  : {roc}')

# roc is around 0.956

from lightgbm import LGBMClassifier
from hyperopt  import fmin, hp, SparkTrials,Trials,tpe ,STATUS_OK
from sklearn.metrics import roc_auc_score , confusion_matrix  
'''
1. objective: binary
2. learning_rate </>
3. max_depth </>
4. n_estimator </>
5. eval_metric
6. scale_pos_weight ( majority / miniority )
7. num_leaves </>
'''
"""
space = {
    'learning_rate': hp.loguniform('learning_rate',np.log(0.0001),np.log(0.01)),
    'max_depth' : hp.quniform('max_detph',20,50,5),
    'n_estimators' : hp.quniform('n_estimator',200,500,45),
    'num_leaves' : hp.quniform('num_leaves',20,35,2)
    }

def trial_objective(params):
    params={
        'boosting_type' : 'gbdt',
        'learning_rate' : params['learning_rate'],
        'max_depth' : int(params['max_depth']),
        'n_estimators' : int(params['n_estimators']),
        'num_leaves' : int(params['num_leaves']),
        # use scale pos weight when you have imblance data +  binary target 
        'scale_pos_weight' : len(x_train_[y_train_ == 0]) / len(x_train_[y_train_ == 1])
    } 
    model = LGBMClassifier(**params)

    model.fit(x_train_,y_train_)
    y_pred = model.predict_proba(x_test_)[:,1]
    def custom_loss(y_tested, y_predicted):
        print(y_tested.shape) 
        print(y_predicted.shape) 
        fn = len(y_predicted[(y_predicted == 0) & (y_tested == 1)])
        tn = len(y_predicted[(y_predicted == 0) & (y_tested == 1)]) 

        diff = tn - fn 
        return np.log(1 + np.exp(-diff))

    return custom_loss(y_test_.to_numpy().flatten(), y_pred) 

best = fmin(trial_objective, space = space, algo =tpe.suggest, max_evals= 40, trials = Trials()) 
best_hyperopt = {'learning_rate': np.float64(0.00010169776502555072), 'max_detph': np.float64(35.0), 'n_estimator': np.float64(225.0), 'num_leaves': np.float64(26.0)}
"""# ///  but that is not fully intelligently


# use optuna of fully intelligently search 
import optuna

def objective(trials): 

    params = {
        'boosting_type': 'gbdt',
        'learning_rate' : trials.suggest_float('learning_rate',0.0001, 0.01), 
        'max_depth' : trials.suggest_int('max_depth',20,40), 
        'n_estimators' : trials.suggest_int('n_estimator',100,400), 
        'num_leaves' : trials.suggest_int('num_leaves',20, 40),
        'scale_pos_weight' : len(y_train_[y_train_ == 1]) / len(y_train_[y_train_ == 0]), 
    }

    model = LGBMClassifier(**params) 
    model.fit(x_train_, y_train_) 
    y_pred = model.predict_proba(x_test_)[:,1]
    def custom_loss(y_tested, y_predicted):
        print(y_tested.shape) 
        print(y_predicted.shape) 
        fn = len(y_predicted[(y_predicted == 0) & (y_tested == 1)])
        tn = len(y_predicted[(y_predicted == 0) & (y_tested == 1)]) 

        diff = tn - fn 
        return np.log(1 + np.exp(-diff))

    return custom_loss(y_test_.to_numpy().flatten(), y_pred) 

study = optuna.create_study(direction = 'maximize')
study.optimize(objective , n_trials = 30)
best_params = study.best_params
print(f'best params : {study.best_params}')
'''
optuna_best : {'learning_rate': 0.0037505617802198942, 'max_depth': 34, 'n_estimator': 219, 'num_leaves': 40}
'''

