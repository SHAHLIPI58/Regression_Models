# -*- coding: utf-8 -*-
"""
Created on Sat May 26 13:45:42 2018

@author: ll
"""

# -*- coding: utf-8 -*-
"""
Created on Sat May 26 12:09:38 2018

@author: ll
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

petrol = pd.read_csv('petrol1.csv')
petrol.head()


dataset = pd.read_csv('petrol1.csv')
y = dataset.iloc[:,0].values
X1 = dataset.iloc[:,2].values

from sklearn.preprocessing import Imputer
imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
X1 = X1.reshape(-1,1)
X1 = imp.fit_transform(X1)
dataset['horsepower'] = X1
X = dataset.iloc[:,1:5].values

plt.figure(figsize=(5,30))
plt.plot(X, y, 'o')
plt.xlabel('temperature')
plt.ylabel('bikes')
plt.show()

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(criterion ="mae",max_depth= 3, min_samples_split= 2,min_samples_leaf = 1)
regressor.fit(X_train,y_train)
y_pred = regressor.predict(X_test)
value_to_predict = [[8,307,307,3504],[8,455,455,4425]]   #(18,14)
regressor .predict(value_to_predict)

def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

dec_tree_rmse=rmse(y_pred,y_test)


# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = regressor, X = X_train, y = y_train, cv = 10)
accuracies.mean()
accuracies.std()


# Applying Grid Search to find the best model and the best parameters
from sklearn.model_selection import GridSearchCV
parameters = [{'min_samples_split': [2,3,4,5], 'min_samples_leaf':[1,2,3],'max_depth':[2,3,4,5,6]}]
grid_search = GridSearchCV(estimator = regressor,
                           param_grid = parameters,
                           scoring = 'r2',
                           cv = 10)
grid_search = grid_search.fit(X_train, y_train)
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_
