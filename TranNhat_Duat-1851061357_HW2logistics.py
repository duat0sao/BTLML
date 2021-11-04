"""
Tran Nhat Duat - 1851061357

         18/03/2000

"""



from __future__ import division, print_function, unicode_literals
import numpy as np 
import matplotlib.pyplot as plt

from perceptronLearningAlgorithm import h
np.random.seed(2)

X = np.array([[5.30, 2.55, 2.80, 3.05, 3.30, 3.55, 3.55, 3.80, 4.05, 4.30, 4.55, 4.80, 5.05, 5.30, 5.80, 6.05, 6.30, 6.55, 6.80, 7.30]])
y = np.array([1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1])

# extended data 
X = np.concatenate((np.ones((1, X.shape[1])), X), axis = 0)



def sigmoid(s):
    return 1/(1 + np.exp(-s))

def logistic_sigmoid_regression(X, y, w_init, eta, tol = 1e-4, max_count = 10000):
    w = [w_init]    
    it = 0
    N = X.shape[1]
    d = X.shape[0]
    count = 0
    check_w_after = 20
    while count < max_count:
        # mix data 
        mix_id = np.random.permutation(N)
        for i in mix_id:
            xi = X[:, i].reshape(d, 1)
            yi = y[i]
            zi = sigmoid(np.dot(w[-1].T, xi))
            w_new = w[-1] + eta*(yi - zi)*xi
            count += 1
            # stopping criteria
            if count%check_w_after == 0:                
                if np.linalg.norm(w_new - w[-check_w_after]) < tol:
                    return w
            w.append(w_new)
    return w
eta = .05 
d = X.shape[0]
w_init = np.random.randn(d, 1)

w = logistic_sigmoid_regression(X, y, w_init, eta)
print(w[-1])



print(sigmoid(np.dot(w[-1].T, X)))







X0 = X[1, np.where(y == 0)][0]
y0 = y[np.where(y == 0)]
X1 = X[1, np.where(y == 1)][0]
y1 = y[np.where(y == 1)]

plt.plot(X0, y0, 'ro' )
plt.plot(X1, y1, 'bs')

xx = np.linspace(2.00, 7.75)
w0 = w[-1][0][0]
w1 = w[-1][1][0]
threshold = -w0/w1
yy = sigmoid(w0 + w1*xx)
plt.axis([1, 9, -1, 2])
plt.plot(xx, yy)
plt.plot(threshold, .5, 'y^')
plt.xlabel('studying hours')
plt.ylabel('predicted probability of pass')
plt.show()






XXX=np.array([2.87, 1.95, 2.53, 2.25, 3.50, 4.75, 1.15, 7.00])
returnKq=np.empty((XXX.shape[0], 1))
for i in range(XXX.shape[0]):
    f=w0 + w1*XXX[i]
    print(f)
    if f>=0.5:
        returnKq[i]=1
    else:
        returnKq[i]=0


print(returnKq)
