from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split 
# Load the dataset
data = fetch_california_housing()

# Features (X) and Target (y)
x = data.data        # input features
y = data.target      # target (house prices)\ 
x_train, x_test, y_train, y_test = train_test_split(x[:4000,:],y[:4000],test_size = 0.2, random_state = 42)

import numpy as np
from sklearn.tree import DecisionTreeRegressor
import polars 

class boosting_model:
    def __init__(self,n_estimators = 500,max_depth = 8, learning_rate = 0.001 , clip = 1 ):
        assert learning_rate is not None , 'warning learning_rate != None '
        self.n_estimators = n_estimators
        self.max_depth = max_depth

        self.learning_rate = learning_rate
        
        self.clip = clip
        self.trees = {} 

    def best_lr(self,gradient,y_pred):

        lr = np.arange(0.001,0.01,step = 0.001)
        min_error = np.inf
        best_index = None
        current_y_pred = y_pred.copy()

        for i, learning_rate in enumerate(lr):
            lr_predict_y_pred =  current_y_pred -  learning_rate * gradient

            error = np.mean((lr_predict_y_pred - y_pred)**2)

            if error < min_error:

                min_error = error
                best_index = i
        return lr[best_index]

    def train(self,x,y):

        self.init_pred_of_train = np.mean(y)
        y_pred_temp = np.full_like(y,self.init_pred_of_train)

        gradient = 2 * ( y_pred_temp - y)
        prev_pred = np.zeros_like(y_pred_temp)
        patience = 0
        min_changes = np.full_like(y_pred_temp,0.0001)
        for i in range(0,self.n_estimators):
            if patience == 10:
                break

            idx = np.random.choice(len(x),size = int(len(x) * 0.8))
            gradient = np.clip(gradient,-self.clip, self.clip)
            model = DecisionTreeRegressor(max_depth =self.max_depth)
            tree_pred_gradient = model.fit(x[idx],gradient[idx]).predict(x)


            y_pred_temp -= self.learning_rate * tree_pred_gradient

            if np.all(np.abs(y_pred_temp - prev_pred) <= min_changes):
                patience += 1

            prev_pred = y_pred_temp.copy()
            gradient = 2 * (y_pred_temp - y)
            self.trees[f'tree_{i}'] = model

    def predict(self,x_test):
        if not isinstance(self.trees,dict): 
            raise RuntimeError('model trees is not found as a {dict} ' )
    
        if (self.init_pred_of_train is None) or ( not self.trees) :  
            raise RuntimeError('Model is not trained yet ( no tree is found)') 
            

        print(f' prediction is run with {len(self.trees)}')
        
        init_pred = self.init_pred_of_train.copy()
        predict_array  = np.full(x_test.shape[0],init_pred)
        for i in range(len(self.trees)):
            updates = self.trees[f'tree_{i}'].predict(x_test)
            predict_array -= self.learning_rate * updates

        return predict_array


model = boosting_model()
model.train(x_train,y_train)
y_pred= model.predict(x_test)
print(np.mean((y_pred - y_test)**2))    
''' 
 prediction is run with 500
0.4994471887199032
''' 

