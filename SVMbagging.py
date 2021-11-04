import numpy as np
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.ensemble import BaggingClassifier

X1 = [[9.37319011,8.71875981], [8.51261889,8.40558943], [9.4696794,8.02144973], [8.78736889,8.29380961], [8.81231157, 8.56119497], [9.03717355,8.93397133], [8.53790057,8.87434722], [9.29312867,9.76537389], [8.38805594,8.86419379], [8.57279694, 7.9070734]]
y1 = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
X2 = [[10.42746579,7.71254431], [11.24760864,9.39846497], [10.33595491,8.61731637], [10.69420104,8.94273986], [11.53897645,9.54957308], [10.3071994,7.19362396], [11.13924705,9.09561534], [11.47383468,9.41269466], [11.00512009,8.89290099], [11.28205624,8.79675607]]
y2 = [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
X = np.array(X1 + X2)
y = y1 + y2



clf = SVC(kernel='linear', C=1E10)
clf.fit(X, y)
#print(clf.support_vectors_)



def plot_svc_decision_function(clf, ax=None, plot_support=True):
    """Plot the decision function for a 2D SVC"""
    if ax is None:
        ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # create grid to evaluate model
    x = np.linspace(xlim[0], xlim[1], 30)
    y = np.linspace(ylim[0], ylim[1], 30)
    Y, X = np.meshgrid(y, x)
    xy = np.vstack([X.ravel(), Y.ravel()]).T
    P = clf.decision_function(xy).reshape(X.shape)

    # plot decision boundary and margins
    ax.contour(X, Y, P, colors='k',
               levels=[-1, 0, 1], alpha=0.5,
               linestyles=['--', '-', '--'])

    # plot support vectors
    if plot_support:
        ax.scatter(clf.support_vectors_[:, 0],
                   clf.support_vectors_[:, 1],
                   s=300, linewidth=1, facecolors='none');
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)


plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='brg');
plot_svc_decision_function(clf)
plt.show()



print('du doan: ', clf.predict([[5.28205624,2.79675607]]))

