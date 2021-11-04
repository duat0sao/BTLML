#Truong Viet Thang - 1851061654


from __future__ import division, print_function, unicode_literals
import numpy as np 
import matplotlib.pyplot as plt
from sklearn import linear_model
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import mean_squared_error
import pandas as pd


# chi phi quang cao (trieu VND)
X0 = np.array([1.0, 5.0, 8.0, 6.0, 3.0, 10.0, 9.0])
# dan so (nghin nguoi)
X1 = np.array([200, 700, 800, 400, 100, 600, 550])
# X
X = np.array([X0, X1]).T

# Doanh thu (trieu VND)
y = np.array([[100, 300, 400, 200, 100, 400, 300 ]]).T

print("X = ", X)

#Mo phong

#fig = plt.figure()
#ax = Axes3D(fig)
#ax.scatter(X0, X1, y, color='#ef1234')
#plt.show()

LR = linear_model.LinearRegression()
LR.fit(X,y)


Yp = LR.predict(X)
print("Y = ", y.T)
print("Yp = ", np.int16(Yp.T))
print("Doanh thu du doan la: ", LR.predict([[7.0, 680]]))
