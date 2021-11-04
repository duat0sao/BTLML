import numpy as np
def tinhdaoham(x):
    return 2*x + 5 * np.cos(x)

def hamfx(x):
    return x*x + 5  * np.sin(x)

def GD(x0, tocdohoc):
    x= [x0]

    for i in range(1000):
        x_new = x[-1] - tocdohoc * tinhdaoham(x[-1])
        if abs(tinhdaoham(x_new)) < 1e-4:
            break
        x.append(x_new)
    return (x,i)


(x,i) = GD(5,0.3)

print('ket qua x cuc tieu = %f, f(x)= %f và f(x) dat cuc tieu sau %d lan lap' %(x[-1],hamfx(x[-1]),i))