#.............Ho Ten : Tran Nhat Duat ..................
#.............MSV    : 1851061357 ......................


from __future__ import print_function
from tkinter import messagebox
import numpy as np 
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from tkinter import *
import tkinter
import pandas as pd
from cvxopt import matrix, solvers
from tkinter import filedialog
from tkinter import Menu
from os import path


#nhap data
N = 10
X0 = np.array([[8.31948309,8.88537088],
[9.72107039,9.44652357],
[9.30445446,8.97514052],
[9.37282942,9.04057585],
[8.99157377,9.10341644],
[9.45728806,9.32995121],
[8.29949031,8.22041161],
[10.1909308,9.84534388],
[8.95613177,8.86582452],
[9.59862883,8.89063922]
])
X1 = np.array([[10.98972272,9.55993622],
[10.14993386,8.25455673],
[10.85817078,9.141431],
[10.82437996,8.69765848],
[11.20625753,9.35678552],
[11.26741918,8.63333018],
[10.91275487,8.62259784],
[11.08286367,9.5424809],
[10.34537123,8.3582377],
[10.08186664,8.70978459]
])

X = np.concatenate((X0.T, X1.T), axis = 1) # all data 
y = np.concatenate((np.ones((1, N)), -1*np.ones((1, N))), axis = 1) # labels


#Tìm lambda
V = np.concatenate((X0.T, -X1.T), axis = 1)
K = matrix(V.T.dot(V))  # see definition of V, K near eq (8)
p = matrix(-np.ones((2*N, 1)))  # all-one vector 

# build A, b, G, h 
G = matrix(-np.eye(2*N))  # for all lambda_n >= 0
h = matrix(np.zeros((2*N, 1)))
A = matrix(y)      # the equality constrain is actually y^T lambda = 0
b = matrix(np.zeros((1, 1))) 
solvers.options['show_progress'] = False
sol = solvers.qp(K, p, G, h, A, b)

l = np.array(sol['x'])
#print('lambda = ')
#print(l.T)

epsilon = 1e-6   # just a small number, greater than 1e-9
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


#Giao dien
windown=Tk()
windown.title("SVM")
windown.geometry("300x200")

#nhap x
label=Label(windown,text="X=",fg="green",font=("arial",13))
label.grid(column=0,row=1)
txt=Entry(windown)
txt.grid(column=1,row=1)


#nhap y
label1=Label(windown,text="Y=",fg="green",font=("arial",13))
label1.grid(column=0,row=2)
txt1=Entry(windown)
txt1.grid(column=1,row=2)

#label ket qua
label2=Label(windown,text="Kết quả: ",fg="green",font=("arial",13))
label2.grid(column=1,row=3)
labelkq=Label(windown,text="",fg="green",font=("arial",18))
labelkq.grid(column=2,row=3)

menu = Menu(windown)
 
new_item = Menu(menu)
 
new_item.add_command(label='New')
 
new_item.add_separator()
 
new_item.add_command(label='Edit')
 
menu.add_cascade(label='File', menu=new_item)
 
windown.config(menu=menu)

#label do chinh xac
label2=Label(windown,text="Độ chính xác: ",fg="green",font=("arial",13))
label2.grid(column=1,row=4)
labelcx=Label(windown,text="",fg="green",font=("arial",13))
labelcx.grid(column=2,row=4)



#file = filedialog.askopenfilename(initialdir= path.dirname(__file__))




#function lam viec khi bam buttonz
def handleButton():
    if(txt.get()!='' and txt1.get()!=''):
        x1_test=float(txt.get())                       #lay du lieu từ textbox1(entry)
        x2_test=float(txt1.get())                      #lay du lieu từ textbox1(entry)
        X_test = [x1_test, x2_test] 
        
        kq = np.sign(np.dot(w.T,np.array(X_test))+b)   #du doan nhan voi Xtest
        
        """
        dot sẽ tính vecto w.T với X_test ta tạo dc ở dòng 105
        được giá trị WTXtest 
        ta cộng WTXtest tính được kia cho b đã tìm được ở dòng 71
        được giá trị class(Xtest)
        và tiếp theo ta dùng sign để xác định dấu class(Xtest) kia
        sign sẽ trả về -1 nếu kết quả kia <0 và if ==0 or >0 thì  return sign sẽ là 1
        """

        y_res = np.sign(np.dot(w.T,np.array(X))+b)
        cx= 100 - np.mean(np.abs(y_res - y)) * 100    #đánh giá mức độ chính xác mô hình 

        labelkq.configure(text=str(kq))  #trả kết quả về dòng labelkq
        labelcx.configure(text=str(cx))  #Trả kết quả về dòng labelcx
    else:
        messagebox.showwarning('Thiếu dữ liệu')
    return

#button
button=Button(windown,text="Du doan",command=handleButton)   
button.grid(column=1,row=5)   


windown.mainloop()



