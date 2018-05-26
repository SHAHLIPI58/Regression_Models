# -*- coding: utf-8 -*-
"""
Created on Sat May 26 14:59:37 2018

@author: ll
"""

# Random Forest Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Importing the dataset
dataset = pd.read_csv('petrol1.csv')
y = dataset.iloc[:,0].values
X1 = dataset.iloc[:,2].values


from sklearn.preprocessing import Imputer
imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
X1 = X1.reshape(-1,1)
X1 = imp.fit_transform(X1)
dataset['horsepower'] = X1
X = dataset.iloc[:,1:5].values

# Splitting the dataset into the Training set and Test set
"""from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)"""

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

# Fitting Random Forest Regression to the dataset
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(max_depth =3,max_features = "log2",min_samples_split =2,
                                  n_estimators = 100, verbose = 0)
regressor.fit(X, y)
important_feature = regressor.feature_importances_
y_random_forest_pred = regressor.predict(X)
# Predicting a new result
y_pred = regressor.predict([[8,307,307,3504],[8,455,455,4425]])

from sklearn.metrics import mean_squared_error
from math import sqrt

rms_svr = sqrt(mean_squared_error(y , y_random_forest_pred))

# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = regressor, X = X, y = y, cv = 10)
accuracies_mean = accuracies.mean()
accuracies.std()

# Applying Grid Search to find the best model and the best parameters
from sklearn.model_selection import GridSearchCV
parameters = [{'min_samples_split':[2,3,4,5],'n_estimators': [70,80,90,100],'verbose':[0], 'max_features':["log2"],
               'max_depth':[1,2,3,4]}]
grid_search = GridSearchCV(estimator = regressor,
                           param_grid = parameters,
                           scoring = 'neg_mean_squared_error',
                           cv = 10)
grid_search = grid_search.fit(X, y)
best_neg_mean_squared_error = grid_search.best_score_
best_parameters = grid_search.best_params_


# Visualising the Random Forest Regression results (higher resolution)
#X_grid = np.arange(min(X), max(X), 0.01)
#X_grid = X_grid.reshape((len(X_grid), 1))
#plt.scatter(X, y, color = 'red')
#plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
#plt.title('Truth or Bluff (Random Forest Regression)')
#plt.xlabel('Position level')
#plt.ylabel('Salary')
#plt.show()