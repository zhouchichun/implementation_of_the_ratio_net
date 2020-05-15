import numpy as np
pi=3.141592653
a=0
A=1

b=1
B=2

exact=3.0000
def give_cos(x):
    x=np.reshape(x,[len(x)])
    ret=np.cos(x+pi/2)
    ret=np.reshape(ret,[-1,1])
    return ret
def give_real(x):
    x=np.reshape(x,[len(x)])
    y=-2/(x+1)+3#np.sin(1)-x
    return y
if __name__=="__main__":
    import matplotlib.pyplot as plt 
    x=[[xx/100.0]for xx in range(1000)]
    y=give_y(x)
    plt.plot(y)
    plt.show()
