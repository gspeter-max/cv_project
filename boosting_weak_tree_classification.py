
from sklearn.datasets import load_breast_cancer

data = load_breast_cancer()
x = data.data    
y = data.target  

from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np 

sss = StratifiedShuffleSplit(n_splits = 1, test_size=0.2,random_state=42)
train_idx , test_idx = next(sss.split(x, y))
x_train, x_test = x[train_idx], x[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

train_val_idx , test_val_idx = next(sss.split(x_train, y_train))
x_train, x_val = x[train_val_idx], x[test_val_idx]
y_train, y_val = y[train_val_idx], y[test_val_idx]


class boosting_classification:

    def __init__(self,learning_rate = 0.001,max_depth = 7 , clip= 1, n_estimators = 200):
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.clip = clip
        self.n_estimators = n_estimators
        self.trees = {} 

    def sigmoid(self,x): 
        return 1/ ( 1 + np.exp(-x))

    def train(self,x,y):
        
        y_pred = np.zeros_like(y,dtype = np.float64)
        gradient = y_pred - y
        patience = 0 
        update_limit = np.full(y_pred.shape[0],0.0001)
        prv_prediction = y_pred.copy()

        for i in range(self.n_estimators): 
            
            if patience == 10: 
                break 
            
            idx = np.random.choice(len(x),size = int(len(x)*0.8))
            model = DecisionTreeRegressor(max_depth=self.max_depth)
            model.fit(x[idx,:],gradient[idx])
            tree_prediction = model.predict(x)
            y_pred -= self.learning_rate * tree_prediction 
            
            if np.all(np.abs(y_pred - prv_prediction) < update_limit):
                patience += 1 

            gradient = self.sigmoid(y_pred) - y 
            self.trees[f'tree_{i}'] = model 
#
    def predict(self,x_test ,threshold = 0.5,required_prob = False): 
        
        if self.trees: 
            if not isinstance(self.trees,dict):
                raise RuntimeError(f"tree is aspect as <class 'dict'> not {type(self.trees)}")

        if  not self.trees: 
            raise RuntimeError('model is not trained yet ')


        logist_array = pre_initial_predict = np.full(x_test.shape[0],0,dtype = np.float64) 
        
        for i in range(len(self.trees)):     

            logist_array -= self.learning_rate * self.trees[f'tree_{i}'].predict(x_test)
        
        prob_array = self.sigmoid(logist_array)
        if required_prob: 
            return prob_array
        predicted_array = np.where((prob_array > threshold), 1,0)
        return predicted_array

model = boosting_classification()
model.train(x_train,y_train)
y_pred = model.predict(x_test)
y_val_pred = model.predict(x_val)

def confusion_matrix(y_pred,y_test):
    
    tp = np.sum((y_pred == 1) & (y_test ==1)) 
    fp = np.sum((y_pred ==1) &   (y_test == 0))
    tn = np.sum((y_pred == 0) & (y_test == 0))
    fn = np.sum((y_pred == 0) & (y_test == 1))

    print(f'''
        [tn : {tn}  fn : {fn}]
        [fp : {fp}  tp : {tp}]
    ''')
confusion_matrix(y_pred,y_test)
confusion_matrix(y_val_pred,y_val)
''' 
performace (test , val)
        
        [tn : 41  fn : 0]
        [fp : 1  tp : 72]
    

        [tn : 35  fn : 0]
        [fp : 1  tp : 55]
    
    ''' 
