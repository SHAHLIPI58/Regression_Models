# -*- coding: utf-8 -*-
"""
Created on Thu May 17 21:33:16 2018

@author: ll
"""

# Multiple Linear Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Importing the dataset
dataset = pd.read_csv('petrol1.csv')
y = dataset.iloc[:,0].values
X = dataset.iloc[:,2].values

from sklearn.preprocessing import Imputer
imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
X = imp.fit_transform(X)
dataset['horsepower'] = X
X1 = dataset.iloc[:,1:5].values


from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X1, y, test_size = 0.2, random_state = 0)

# Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)


#root mean square error
from sklearn.metrics import mean_squared_error
from math import sqrt

rms = sqrt(mean_squared_error(y_test, y_pred))



#backword elimination regression
import statsmodels.formula.api as sm
X1 = np.append(arr = np.ones((398,1)).astype(int), values = X1, axis = 1)
X_opt= X1
#X_opt = X1[:,[0,1,2,3,4]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()





from sklearn.cross_validation import train_test_split
X_train_opt, X_test_opt, y_train_opt, y_test_opt = train_test_split(X_opt, y, test_size = 0.2, random_state = 0)

# Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor1 = LinearRegression()
regressor1.fit(X_train_opt, y_train_opt)

y_pred_opt = regressor1.predict(X_test_opt)


from sklearn.metrics import mean_squared_error
from math import sqrt

rms_opt = sqrt(mean_squared_error(y_test_opt, y_pred_opt))



