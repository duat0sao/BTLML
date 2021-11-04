"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#numOfPoint = 30
#noise = np.random.normal(0,1,numOfPoint).reshape(-1,1)
#x = np.linspace(30, 100, numOfPoint).reshape(-1,1)
#N = x.shape[0]
#y = 15*x + 8 + 20*noise
#plt.scatter(x, y)

data = pd.read_csv('data_linear.csv').values
N = data.shape[0]
x = data[:, 0].reshape(-1, 1)
y = data[:, 1].reshape(-1, 1)
plt.scatter(x, y)
plt.xlabel('mét vuông')
plt.ylabel('giá')

x = np.hstack((np.ones((N, 1)), x))

w = np.array([0.,1.]).reshape(-1,1)

numOfIteration = 100
cost = np.zeros((numOfIteration,1))
learning_rate = 0.000001
for i in range(1, numOfIteration):
    r = np.dot(x, w) - y
    cost[i] = 0.5*np.sum(r*r)
    w[0] -= learning_rate*np.sum(r)
    # correct the shape dimension
    w[1] -= learning_rate*np.sum(np.multiply(r, x[:,1].reshape(-1,1)))
    print(cost[i])
predict = np.dot(x, w)
plt.plot((x[0][1], x[N-1][1]),(predict[0], predict[N-1]), 'r')
plt.show()

x1 = 50
y1 = w[0] + w[1] * 20
print('Giá nhà cho 20m^2 là : ', y1)

"""



"""

from __future__ import division, print_function, unicode_literals
import numpy as np 
import matplotlib.pyplot as plt

# dien tich
X = np.array([[30, 32, 34, 37, 39]]).T
# gia tien
y = np.array([[ 448524, 509248, 535104, 551432, 632418]]).T
# Visualize data 
plt.plot(X, y, 'ro')
plt.axis([25, 120, 400000, 1000000]) #chi so cua bieu do x,x,y,y
plt.xlabel('m2')
plt.ylabel('VND')
plt.show()


# Building Xbar 
one = np.ones((X.shape[0], 1))
Xbar = np.concatenate((one, X), axis = 1)

# Calculating weights of the fitting line 
A = np.dot(Xbar.T, Xbar)
b = np.dot(Xbar.T, y)
w = np.dot(np.linalg.pinv(A), b)
print('w = ', w)
# Preparing the fitting line 
w_0 = w[0][0]
w_1 = w[1][0]
x0 = np.linspace(30, 39)
y0 = w_0 + w_1*x0

# Drawing the fitting line 
plt.plot(X.T, y.T, 'ro')     # data 
plt.plot(x0, y0)               # the fitting line
plt.axis([25, 120, 400000, 1000000])    #chi so cua bieu do 
plt.xlabel('m2')       
plt.ylabel('VND')
plt.show()


y1 = w_1*357 + w_0
#y2 = w_1*160 + w_0

print( 'gia can nha 357m2 la: %.2f '  %(y1) )
#print( u'Predict weight of person with height 160 cm: %.2f (kg), real number: 56 (kg)'  %(y2) )
"""


"""
from sklearn import datasets, linear_model

# fit the model by Linear Regression
regr = linear_model.LinearRegression(fit_intercept=False) # fit_intercept = False for calculating the bias
regr.fit(Xbar, y)

# Compare two results
print( 'Solution found by scikit-learn  : ', regr.coef_ )
print( 'Solution found by (5): ', w.T)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

data = pd.read_csv('data_linear.csv')
print(data.shape)
data.head()

chi = np.array([1.0, 5.0, 8.0, 6.0, 3.0, 10.0, 9.0])
dan = np.array([200, 700, 800, 400, 100, 600, 550])
doanh = np.array([100, 300, 400, 200, 100, 400, 300 ])

# Ploting the scores as scatter plot
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(chi, dan, doanh, color='#ef1234')
plt.show()

X = np.array([chi, dan]).T
Y = np.array(doanh)

# Model Intialization
reg = LinearRegression()
# Data Fitting
reg = reg.fit(X, Y)

# Y Prediction
Y_pred = reg.predict(X)

# Model Evaluation
rmse = np.sqrt(mean_squared_error(Y, Y_pred))
r2 = reg.score(X, Y)



print("Y = ", Y.T)
print("Yp = ", np.int16(Y_pred.T))

print("Doanh thu du doan la: ", reg.predict([[7.0, 680]]))







