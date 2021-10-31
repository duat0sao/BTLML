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
import cv2
from sklearn.metrics import accuracy_score




#  nhập dữ liệu
Categories=['Mắt lờ đờ','Mắt mở','Mắt nhắm']
flat_data_arr=[]                                            #input array
target_arr=[]                                               #output array
datadir='C:/xampp/htdocs/hoc/ML/BTL/dataset'
# nhập từng dữ liệu
for i in Categories:
    
    print(f'loading... category : {i}')
    path=os.path.join(datadir,i)
    for img in os.listdir(path):
        
        img_array=imread(os.path.join(path,img))
        img_resized=resize(img_array,(150,150,3))
        flat_data_arr.append(img_resized.flatten())
        target_arr.append(Categories.index(i))
    print(f'Done!!!')
flat_data=np.array(flat_data_arr)
target=np.array(target_arr)


df=pd.DataFrame(flat_data)                                  #dataframe
df['Target']=target
x=df.iloc[:,:-1]                                            #input data 
y=df.iloc[:,-1]                                             #output data



# Tách dữ liệu phân tích // phân tích dữ liệu
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.10,random_state=77,stratify=y)



# SVM
print('SVM')
model=svm.SVC(probability=True)
print('Model is training..............')
model.fit(x_train,y_train)
print('Done!!!\n')


y_pred=model.predict(x_test)
print("Kết quả dự đoán :")
print(y_pred)
print("Kết quả thực tế:")
print(np.array(y_test))
print(f"Độ chính xác: {accuracy_score(y_pred,y_test)*100}% ")



"""
print('\n#####-------- Prediction on single image ------#####')
new_img = 'C:/xampp/htdocs/hoc/ML/BTL/Eyes/Closed/10.jpg'
img=imread(new_img)
plt.imshow(img)
#plt.show()

img_resize=resize(img,(150,150,3))
l=[img_resize.flatten()]
probability=model.predict_proba(l)
print("The predicted image is : "+Categories[model.predict(l)[0]])
"""



#Giao dien
windown=Tk()
windown.title("SVM")
windown.geometry("300x250")

#nhap anh 
label=Label(windown,text="Nhap anh: ",fg="green",font=("arial",13))
label.grid(column=0,row=1)

txt= filedialog.askopenfilename()



#label ket qua
label2=Label(windown,text="Kết quả: ",fg="green",font=("arial",13))
label2.grid(column=1,row=3)
labelkq=Label(windown,text="",fg="green",font=("arial",18))
labelkq.grid(column=2,row=3)




#function lam viec khi bam buttonz
def handleButton():
    if(txt!=''):

        img=imread(txt)
        plt.imshow(img)
        

        img_resize=resize(img,(150,150,3))
        l=[img_resize.flatten()]
        probability=model.predict_proba(l)
        kq = Categories[model.predict(l)[0]]

        labelkq.configure(text=str(kq))  #trả kết quả về dòng labelkq


    return

#button
button=Button(windown,text="Dự đoán",command=handleButton)   
button.grid(column=1,row=5)   

 
    

windown.mainloop()



