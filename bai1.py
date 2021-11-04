# Tran Nhat Duat - msv:1851061357

"""
from __future__ import division, print_function, unicode_literals
import numpy as np 
import matplotlib.pyplot as plt


# dien tich
X = np.array([[48, 50.4138, 52.8276, 55.2414, 57.6552, 40.069, 62.8966, 64.8966, 67.3103, 69.7241, 71.1379, 74.5517, 76.3793, 79.3793, 81.7931, 83.2069, 85.6207, 88.0345, 90.4483, 92.8621, 95.2749, 97.6897, 100.1034, 102.5172, 105.931, 108.3448, 110.7586, 113.1724, 115.5863, 118]]).T
# gia tien
y = np.array([[ 3448.524, 3509.248, 3535.104, 3552.432, 3623.418, 3645.992, 3655.248, 3701.377, 3746.918, 3757.881, 3831.004, 3855.183, 3888.707, 3902.545, 3952.545, 3999.531, 4069.78, 4074.42, 4103.88, 4139.99, 4153.13, 4240.70, 4251.99, 4287.97, 4320.99, 4374.44, 4444.44, 4469.69, 4478.98, 4545.56]]).T




plt.plot(X, y, 'ro')
plt.axis([40, 130, 3000, 4550])     #chi so cua bieu do x,x,y,y
plt.xlabel('m2')
plt.ylabel('ty? VND')
plt.show()



one = np.ones((X.shape[0], 1))
Xbar = np.concatenate((one, X), axis = 1)



A = np.dot(Xbar.T, Xbar)
b = np.dot(Xbar.T, y)
w = np.dot(np.linalg.pinv(A), b)
print('w = ', w)



w_0 = w[0][0]
w_1 = w[1][0]
x0 = np.linspace(48, 118) 
y0 = w_0 + w_1*x0




plt.plot(X.T, y.T, 'ro')          # data 
plt.plot(x0, y0)                  # duong
plt.axis([40, 130, 3000, 4550])   #chi so cua bieu do x,x,y,y
plt.xlabel('m2')       
plt.ylabel(' ty? VND')
plt.show()


y1 = w_1*357 + w_0


print( u'Gia can nha 357m2 la: %.2f ty? VND'  %(y1) )





#.............Ho Ten : Tran Nhat Duat ..................
#.............MSV    : 1851061357 .....................
from tkinter import *
import tkinter

from sklearn.svm import SVC
from sklearn.utils import resample
import pandas as pd

data = pd.read_csv("SVM.csv", header=None, usecols=[i for i in range(3)])
print(data)
# the usecols=[i for i in range(11)] will create a list of numbers for your columns
# that line will make a dataframe called data, which will contain your data.
l = [i for i in range(2)]
X_train = data[l]
y_train = data[2]

clf = SVC(kernel = 'linear', C = 1e5) # just a big number
clf.fit(X_train, y_train ) # each sample is one row

w = clf.coef_
b = clf.intercept_
print('w = ', w)
print('b = ', b)

predict= pd.read_csv("SVM.csv", header=None, usecols=[i for i in range(2)])
l1 = [i for i in range(2)]
X_pre = predict[l]


Z = clf.predict(X_pre)

# bộ phân lớp 1
boot1=resample(data,replace=True,n_samples=20,random_state=1)
l = [i for i in range(2)]
X_train = boot1[l]
y_train = boot1[2]

clf1 = SVC(kernel = 'linear', C = 1e5)
clf1.fit(X_train, y_train )

# bộ phân lớp 2
boot2=resample(data,replace=True,n_samples=20,random_state=1)
l = [i for i in range(2)]
X_train = boot2[l]
y_train = boot2[2]

clf2 = SVC(kernel = 'linear', C = 1e5)
clf2.fit(X_train, y_train )

# bộ phân lớp 3
boot3=resample(data,replace=True,n_samples=20,random_state=1)
l = [i for i in range(2)]
X_train = boot3[l]
y_train = boot3[2]

clf3 = SVC(kernel = 'linear', C = 1e5)
clf3.fit(X_train, y_train )




# boot = resample(data, replace=True, n_samples=4, random_state=1)

windown=Tk()
windown.title("SVM")
#Nhập x
label=Label(windown,text="X=",fg="green",font=("arial",20))
label.grid(column=0,row=1)
txt=Entry(windown)
txt.grid(column=1,row=1)

#Nhập y
label1=Label(windown,text="Y=",fg="green",font=("arial",20))
label1.grid(column=0,row=2)
txt1=Entry(windown)
txt1.grid(column=1,row=2)

#Nhãn dự đoán
label2=Label(windown,text="Nhãn dự đoán là:",fg="green",font=("arial",20))
label2.grid(column=0,row=3)
labelkq=Label(windown,text="",fg="green",font=("arial",20))
labelkq.grid(column=1,row=3)

#button
def handleButton():
    Z=0.0
    Z1=0.0
    Z2=0.0
    Z3 = 0.0
    if(txt.get()!='' and txt1.get()!=''):
        a=float(txt.get())
        b=float(txt1.get())
    Z1 = clf1.predict([[a,b]])
    Z2= clf2.predict([[a,b]])
    Z3= clf3.predict([[a,b]])
    if(Z1==Z2):
        Z=Z1
    if (Z2 == Z3):
        Z = Z2
    if (Z1 == Z3):
        Z = Z1
    labelkq.configure(text=str(Z))
    return
button=Button(windown,text="Dự Đoán Nhãn",command=handleButton)
button.grid(column=0,row=4)

windown.mainloop()

"""

"""
# Import các thư viện 
from __future__ import print_function
import numpy as np 
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from cvxopt import matrix, solvers
np.random.seed(22)

means = [[2, 2], [4, 2]]
cov = [[.3, .2], [.2, .3]]

# Tạo data cho 2 lables
N = 10
X0 = np.random.multivariate_normal(means[0], cov, N) # class 1
X1 = np.random.multivariate_normal(means[1], cov, N) # class -1 
X = np.concatenate((X0.T, X1.T), axis = 1)

y = np.concatenate((np.ones((1, N)), -1*np.ones((1, N))), axis = 1) # labels
print(X)
print(y)
#Tìm lambda
V = np.concatenate((X0.T, -X1.T), axis = 1)
K = matrix(V.T.dot(V)) 
p = matrix(-np.ones((2*N, 1))) 
G = matrix(-np.eye(2*N)) 
h = matrix(np.zeros((2*N, 1)))
A = matrix(y) 
b = matrix(np.zeros((1, 1))) 
solvers.options['show_progress'] = False
sol = solvers.qp(K, p, G, h, A, b)
l = np.array(sol['x'])
print('lambda = ')
print(l.T)

epsilon = 1e-6 
S = np.where(l > epsilon)[0]
VS= V[:, S]
XS = X[:, S]
yS = y[:, S]
lS = l[S]

# Tính toán hệ số w và b :
w = VS.dot(lS)
b = np.mean(yS.T - w.T.dot(XS))
print('w = ', w.T)
print('b = ', b)
#Test 6 point

X_test = [11,8]

label = np.sign(np.dot(w.T,np.array(X_test))+b)

#print(X_test)
print(label)

"""

import pandas as pd
import os
from skimage.transform import resize
from skimage.io import imread
import numpy as np
import matplotlib.pyplot as plt
from tkinter import *
import tkinter
from sklearn.model_selection import train_test_split
from tkinter import filedialog
from sklearn import svm
from sklearn.metrics import accuracy_score
from tkinter import messagebox
from PIL import ImageTk, Image



flat_data_arr=[]                                            #khởi tạo output array
datadir='C:/xampp/htdocs/hoc/ML/BTL/dataset/Mắt lờ đờ'



    
   
img_array=imread('C:/xampp/htdocs/hoc/ML/BTL/dataset/Mắt lờ đờ/1.jpg')
img_resized=resize(img_array,(150,150,3))          #thay đổi kích thước ảnh
flat_data_arr.append(img_resized.flatten())       
flat_data=np.array(flat_data_arr)
df=pd.DataFrame(flat_data) 

print('flat data arr')               
print(flat_data_arr)
print('flat data')
print(flat_data)
print('df')
print(df)





