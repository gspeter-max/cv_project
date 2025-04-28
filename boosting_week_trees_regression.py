import numpy as np
from sklearn.tree import DecisionTreeRegressor

np.random.seed(42)
x = np.random.randn(50,2)
y = 3 * x[:,1].flatten() + np.random.normal(0, 1, size=50)

class boosting_model: 
    def __init__(self,n_estimators = 500,max_depth = 8, learning_rate = None , clip = 1 ): 
        self.n_estimators = n_estimators 
        self.max_depth = max_depth 
        self.learning_rate = learning_rate 
        self.clip = clip 
    
    def best_lr(self,gradient,y_pred): 

        lr = np.arange(0.001,0.01,step = 0.001)
        min_error = np.inf 
        best_index = None
        current_y_pred = y_pred.copy() 

        for i, learning_rate in enumerate(lr): 
            lr_predict_y_pred =  current_y_pred -  learning_rate * gradient 
            error = np.mean((lr_predict_y_pred - y)**2)
            if error < min_error: 
                min_error = error 
                best_index = i
        return lr[best_index] 
        
    def forward(self,x,y): 
        
        y_pred = np.mean(y)
        y_pred_temp = np.full_like(y,y_pred)
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

            if self.learning_rate is not None: 
                y_pred_temp -=  self.learning_rate * tree_pred_gradient
            else :
                learning_rate = self.best_lr(tree_pred_gradient,y_pred_temp)  
                y_pred_temp -= learning_rate * tree_pred_gradient
                    
            if np.all(np.abs(y_pred_temp - prev_pred) <= min_changes):
                patience += 1
            
            prev_pred = y_pred_temp.copy() 
            gradient = 2 * (y_pred_temp - y)

        return y_pred_temp
model = boosting_model() 
y_pred = model.forward(x,y)
print(y_pred)
