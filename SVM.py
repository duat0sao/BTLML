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




Categories=['Mắt lờ đờ','Mắt mở','Mắt nhắm']                # 1 string
flat_data_arr=[]                                            #khởi tạo input array
target_arr=[]                                               #khởi tạo output array
datadir='C:/xampp/htdocs/hoc/ML/BTL/dataset'

# load dữ liệu
for i in Categories:
    
    print(f'loading...  {i}...')
    path=os.path.join(datadir,i)
    for img in os.listdir(path):
        
        img_array=imread(os.path.join(path,img))
        img_resized=resize(img_array,(150,150,3))          #thay đổi kích thước ảnh
        flat_data_arr.append(img_resized.flatten())
        """

Lớp làm phẳng Keras rất quan trọng khi bạn phải xử lý các đầu vào đa chiều như tập dữ liệu hình ảnh. 
Keras.layers.flattenhàm làm phẳng các bộ căng đầu vào đa chiều thành một chiều duy nhất, vì vậy bạn có thể lập mô hình lớp đầu vào và xây dựng mô hình mạng nơ-ron của mình, 
sau đó chuyển các dữ liệu đó vào từng nơ-ron của mô hình một cách hiệu quả.
Bạn có thể hiểu điều này dễ dàng với tập dữ liệu MNIST thời trang. 
Hình ảnh trong tập dữ liệu này có kích thước 28 * 28 pixel. 
Do đó, nếu bạn in hình ảnh đầu tiên trong python, bạn có thể thấy một mảng đa chiều, 
mảng mà chúng tôi thực sự không thể đưa vào lớp đầu vào của Mạng thần kinh sâu của chúng tôi.
Để giải quyết vấn đề này, chúng ta có thể làm phẳng dữ liệu hình ảnh khi đưa nó vào mạng nơ-ron. 
Chúng ta có thể làm điều này bằng cách biến tensor đa chiều này thành mảng một chiều. 
Trong mảng phẳng này bây giờ chúng ta có 784 phần tử (28 * 28). Sau đó, chúng ta có thể tạo ra lớp đầu vào với 784 nơ-ron để xử lý từng phần tử của dữ liệu đến.
Tất cả chúng ta có thể làm điều này bằng cách sử dụng một dòng mã, loại

append là thêm dữ liệu vào phần sau VD x=123 x.append45 => x= 12345
"""

        target_arr.append(Categories.index(i))           #gán nhãn cho tập dữ liệu vừa tạo 0 1 2
    print(f'Done!!!')


flat_data=np.array(flat_data_arr)
target=np.array(target_arr)




df=pd.DataFrame(flat_data)                                  #dataframe
df['Target']=target
x=df.iloc[:,:-1]                                            #input data 
y=df.iloc[:,-1]                                             #output data


# Tách dữ liệu 
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=130,stratify=y)


"""
print(x_train\n)

print(y_train\n)

print(x_test\n)

print(y_test\n)

"""


# ## train SVM 
print('SVM')
model=svm.SVC(probability=True)
print('Model is training..............')
model.fit(x_train,y_train)
print('Done!!!\n')


def cxtt():                                             #hàm tính độ chính xác
    
    y_pred=model.predict(x_test)
    print("Kết quả dự đoán :")
    print(y_pred)
    print("Kết quả thực tế:")
    print(np.array(y_test))
    cxpt = accuracy_score(y_pred,y_test)*100
    print(f"Độ chính xác: {accuracy_score(y_pred,y_test)*100}% ")
    messagebox.showinfo("Độ chính xác: ", cxpt)



#Giao dien
windown=Tk()
windown.title("SVM")
windown.geometry("550x300")


def openfn():
    filename = filedialog.askopenfilename()
    return filename
def open_img():
    anh = openfn()
    img = Image.open(anh)

    img = img.resize((150, 50), Image.ANTIALIAS)
    img = ImageTk.PhotoImage(img)
    panel = Label(windown, image=img)
    panel.image = img
    panel.pack()


    img=imread(anh)
    plt.imshow(img)
    img_resize=resize(img,(150,150,3))
    l=[img_resize.flatten()]

    probability=model.predict_proba(l)
    kq = Categories[model.predict(l)[0]]
    print(kq)
    messagebox.showinfo('Kết quả dự đoán', kq)
    
def clear():                                            #đóng luôn chứ ko phải clear
    windown.quit()             #quit luôn
    #windown.destroy()         #cũng là quit nhưng mà nhẹ hơn




btn = Button(windown, text='open image', command=open_img).pack(side = TOP, fill = BOTH)
cx = Button(windown, text='Độ chính xác thuật toán', command=cxtt).pack(side = TOP, fill = BOTH)
clearScr = Button(windown, text='Đóng', command=clear).pack(side = TOP,  fill = BOTH)

windown.mainloop()



