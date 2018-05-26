# Polynomial Regression

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
X1 = imp.fit_transform(X1)
dataset['horsepower'] = X1
X = dataset.iloc[:,1:5].values

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)"""

# Fitting Linear Regression to the dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
ans_lin= lin_reg.fit(X_train, y_train)
y_lin_pred = ans_lin.predict(X_test)
#predicted value:
lin_predicted_ans = ans_lin.predict([[8,307,307,3504]])


# Fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 2)
X_poly = poly_reg.fit_transform(X_train)
poly_reg.fit(X_poly, y_train)
lin_reg_2 = LinearRegression()
ans_poly = lin_reg_2.fit(X_poly, y_train)
y_poly_pred=ans_poly.predict(poly_reg.fit_transform(X_test))
#predicted value:
poly_predicted_ans=ans_poly.predict( poly_reg.fit_transform([[8,307,307,3504]]))


import statsmodels.formula.api as sm
X1 = np.append(arr = np.ones((398,1)).astype(int), values = X, axis = 1)
X_opt= X1
#X_opt = X1[:,[0,1,2,3,4]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()




import statsmodels.formula.api as sm
def backwardElimination(x, sl):
    numVars = len(x[0])
    for i in range(0, numVars):
        regressor_OLS = sm.OLS(y, x).fit()
        maxVar = max(regressor_OLS.pvalues).astype(float)
        if maxVar > sl:
            for j in range(0, numVars - i):
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                    x = np.delete(x, j, 1)
    regressor_OLS.summary()
    return x
 
SL = 0.005
X_opt = X_opt
X_Modeled = backwardElimination(X_opt, SL)

#devide into test and train model
from sklearn.cross_validation import train_test_split
X_train_opt, X_test_opt, y_train_opt, y_test_opt = train_test_split(X_Modeled, y, test_size = 0.2, random_state = 0)



lin_reg_3 = LinearRegression()
ans_poly3 = lin_reg_3.fit(X_train_opt, y_train_opt)
y_poly_pred_new=ans_poly3.predict(X_test_opt)


from sklearn.metrics import mean_squared_error
from math import sqrt

rms_lin = sqrt(mean_squared_error(y_test, y_lin_pred))
rms_poly = sqrt(mean_squared_error(y_test, y_poly_pred))
rms_poly_back = sqrt(mean_squared_error(y_test_opt, y_poly_pred_new))

#data = np.random.randint(3, 7, (10, 1, 1, 80))
#newdata = np.squeeze(data) # Shape is now: (10, 80)
#plt.plot(newdata) # plotting by columns
#plt.show()


#plt.plot(x11,y11,z11)
#plt.show()
#
#import plotly
#plotly.tools.set_credentials_file(username='LipiShah', api_key='iZSP3aNrpvVzRI9wVeHo')
#import plotly.plotly as py
#import plotly.graph_objs as go
#
## Create random data with numpy
#import numpy as np
#
#N = 100
#random_x = np.linspace(0, 1, N)
#random_y0 = np.random.randn(N)+5
#random_y1 = np.random.randn(N)
#random_y2 = np.random.randn(N)-5
#
## Create traces
#trace0 = go.Scatter(
#    x = y,
#    y = y_lin_pred,
#    mode = 'lines',
#    name = 'lines'
#)
#trace1 = go.Scatter(
#    x = y,
#    y = y_poly_pred,
#    mode = 'lines+markers',
#    name = 'lines+markers'
#)
#trace2 = go.Scatter(
#    x = random_x,
#    y = random_y2,
#    mode = 'markers',
#    name = 'markers'
#)
#data = [trace0,trace1]
#
#py.iplot(data, filename='line-mode')
#
#
## Visualising the Linear Regression results
#plt.scatter(X, y, color = 'red')
#plt.plot(X, lin_reg.predict(X), color = 'blue')
#plt.title('Truth or Bluff (Linear Regression)')
#plt.xlabel('Position level')
#plt.ylabel('Salary')
#plt.show()


# Visualising the Polynomial Regression results
plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg_2.predict(poly_reg.fit_transform(X)), color = 'blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# Visualising the Polynomial Regression results (for higher resolution and smoother curve)
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, lin_reg_2.predict(poly_reg.fit_transform(X_grid)), color = 'blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# Predicting a new result with Linear Regression
lin_reg.predict(6.5)

# Predicting a new result with Polynomial Regression
lin_reg_2.predict(poly_reg.fit_transform(6.5))