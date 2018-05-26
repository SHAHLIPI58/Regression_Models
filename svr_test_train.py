# SVR

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
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X_train = sc_X.fit_transform(X_train)
y_train = y_train.reshape(-1,1) 
y1_train = sc_y.fit_transform(y_train)

# Fitting SVR to the dataset
from sklearn.svm import SVR
regressor = SVR(C=0.5,epsilon = 0.3, gamma = 0.6)
ans = regressor.fit(X_train, y1_train)
pred_SVR = ans.predict(sc_X.fit_transform(X_test))
predicted_y_test = sc_y.inverse_transform(pred_SVR)
regressor.score(X_train, y1_train)

from sklearn.metrics import mean_squared_error
from math import sqrt

rms_svr = sqrt(mean_squared_error(y_test, predicted_y_test))


# Predicting a new result
value_to_predict = [[8,307,307,3504],[8,455,455,4425]]
y_pred = ans.predict(sc_X.fit_transform(value_to_predict))
y_pred1 = sc_y.inverse_transform(y_pred)


# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = regressor, X = X_train, y = y1_train, cv = 10)
accuracy_mean = accuracies.mean()
accuracies.std()


# Applying Grid Search to find the best model and the best parameters
from sklearn.model_selection import GridSearchCV
parameters = [{'C': [1,0.5,0.6,0.7,0.8,0.9,2.0], 'kernel': ['rbf'], 'gamma': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,0.09], 'epsilon' :[0.1,0.2,0.3,1.0,2.0,10.0]}]
grid_search = GridSearchCV(estimator = regressor,
                           param_grid = parameters,
                           scoring = 'neg_mean_squared_error',
                           cv = 10)
grid_search = grid_search.fit(X_train, y1_train)
best_neg_mean_squared_error = grid_search.best_score_
best_parameters = grid_search.best_params_

# Visualising the SVR results
#plt.scatter(X, y, color = 'red')
#plt.plot(X, regressor.predict(X), color = 'blue')
#plt.title('Truth or Bluff (SVR)')
#plt.xlabel('Position level')
#plt.ylabel('Salary')
#plt.show()
#
## Visualising the SVR results (for higher resolution and smoother curve)
#X_grid = np.arange(min(X), max(X), 0.01) # choice of 0.01 instead of 0.1 step because the data is feature scaled
#X_grid = X_grid.reshape((len(X_grid), 1))
#plt.scatter(X, y, color = 'red')
#plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
#plt.title('Truth or Bluff (SVR)')
#plt.xlabel('Position level')
#plt.ylabel('Salary')
#plt.show()