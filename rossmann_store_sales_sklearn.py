'''
target ==> ( num_customer, scales)
'''

import polars

class data_pipeline:

    def __init__(self,_1st_data : str, _2nd_data : str = '/content/drive/MyDrive/store.csv'):
        self._1st_data = _1st_data
        self._2nd_data = _2nd_data

    def forward(self):

        df1 = polars.read_csv(self._1st_data,ignore_errors= True)
        df2= polars.read_csv(self._2nd_data, ignore_errors= True)

        df1 = df1.with_columns(
            polars.col('StateHoliday').fill_null(1)
        )

        df1 = df1.with_columns(
            polars.col('Date').str.to_datetime('%Y-%m-%d')
        )
        df1 = df1.with_columns([
            polars.col('Date').dt.day().alias('day'),
            polars.col('Date').dt.month().alias('month'),
            polars.col('Date').dt.year().alias('year'),
            polars.col('Date').dt.week().alias('Yearofweek')
        ])


        for i, month_values in enumerate(df2['PromoInterval']):
            month_values = month_values.split(',')
            for sub_month_values in month_values:
                if sub_month_values != '':
                    if sub_month_values not in df2.columns:
                        df2 = df2.with_columns(polars.lit(0).alias(sub_month_values))
                    df2[i,sub_month_values] = 1

        df2 = df2.drop('PromoInterval')
        df2 = df2.with_columns(polars.col('CompetitionDistance').log().alias('CompetitionDistance'))

        def one_hot(df,col):
            for i,values in enumerate(df[col]):
                if values not in df.columns:
                    df = df.with_columns(polars.lit(0).alias(values))

                df[i,values] = 1
            return df


        df2 = one_hot(df2,'StoreType')
        df2 = one_hot(df2,'Assortment')
        df2 = df2.drop('StoreType','Assortment')


        final_df= df1.join(
            other = df2,
            on = 'Store',
            how = 'left'
        )

        return final_df

train_df = data_pipeline('/content/drive/MyDrive/train_store_data.csv').forward()
# test_df = data_pipeline('/content/drive/MyDrive/test_rossanmand.csv').forward()
# print(len(train_df.columns))
# print(len(test_df.columns))

def split_data(df,return_x = False):
    x = df.drop('Sales','Store','Date' ,'Customers')
    y = df.select('Sales')
    return x, y

x, y = split_data(train_df)

df_numeric = x.select(polars.col(polars.Int64, polars.Float64,polars.Int32,polars.Float32))

mins = df_numeric.min()
maxs = df_numeric.max()

for columns in df_numeric.columns:
    x = x.with_columns((polars.col(columns) - mins[columns]) / (maxs[columns] - mins[columns]))


x, y = x.to_numpy() , y.to_numpy()

test_df = data_pipeline('/content/drive/MyDrive/test_rossanmand.csv').forward().drop('Id','Store','Date').to_numpy()
# x , y = x[:10000,:].to_numpy(), y[:10000].to_numpy()

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

sns.histplot(y,kde = True) ; plt.show()
sns.histplot(np.log1p(y),kde = True) ; plt.show()
sns.histplot(x[:10000,:],kde = True) ; plt.show()

from sklearn.model_selection import train_test_split
import lightgbm as lgb
import optuna
import numpy as np


x_train, x_val , y_train, y_val = train_test_split(x,y,test_size = 0.2,random_state = 42)

"""

# do taht if you have good machine

def objective(trials):

    params = {
        'n_estimators' : 1000,
        'boosting_type' : trials.suggest_categorical('boosting_type',['dart','gbdt']),
        'objective' : 'regression_l1',
        'max_depth' : trials.suggest_int('max_deptt',7,20),
        'num_leaves' : trials.suggest_int('num_leaves',20,300),
        'learning_rate' : trials.suggest_float('learning_rate',0.001,0.01,log = True),
        'min_child_samples' : trials.suggest_int('min_child_samples',5,100),
        'reg_alpha' : trials.suggest_float('reg_alpha',0.001,10,log = True),
        'reg_lamdba' : trials.suggest_float('reg_lambda',0.001,10,log= True),
        'subsample' : trials.suggest_float('subsample',0.6,1.0),
        'colsample_bytree' : trials.suggest_float('colsample_bytree',0.6,1.0)
    }
    stopping_round = trials.suggest_int('stopping_rounds',10,50)

    model = lgb.LGBMRegressor(**params)
    model.fit(
        x_train,
        np.log1p(y_train),
        eval_set = [(x_val, np.log1p(y_val))],
        callbacks = [
            lgb.early_stopping(stopping_rounds = stopping_round ),
            lgb.log_evaluation(period = 50)
        ]
    )

    y_pred = model.predict(x_val)
    return - np.mean(np.abs(np.expm1(y_pred - y_val)))

create_study = optuna.create_study(direction= 'maximize')
create_study.optimize(objective,n_trials = 20)"""

from sklearn.metrics import mean_absolute_error
# best_params = create_study.best_params

best_params = {'boosting_type': 'gbdt', 'max_depth': -1, 'num_leaves': 200, 'learning_rate': 0.011, 'min_child_samples': 30, 'reg_alpha': 0.08512988788905049, 'reg_lambda': 0.034522768357719315, 'subsample': 0.6852853676304876, 'colsample_bytree': 0.851591132790986, 'stopping_rounds': 16}
stopping_rounds = best_params['stopping_rounds']
del best_params['stopping_rounds']

best_params['n_estimators'] = 5000

best_model = lgb.LGBMRegressor(**best_params)
best_model.fit(
    x_train, np.log1p(y_train),
    eval_set=[(x_val,np.log1p(y_val))],
    callbacks = [
        lgb.early_stopping(stopping_rounds=stopping_rounds),
        lgb.log_evaluation(10)
    ]
)
y1_pred_val = best_model.predict(x_val)
print((np.expm1(y1_pred_val) ,y_val))
print(mean_absolute_error(y_val ,np.expm1(y1_pred_val)))

y_pred = np.expm1(best_model.predict(test_df)) 
print(y_pred)
