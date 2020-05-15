

def a(i):
    return 2.0-1.0/i
def b(i):
    return 1.0-1.0/i

def give_legend(x,n):
    ret={1:1,2:x}
    for i in range(3,n+1):
        ret[i]=a(i-1)*x*ret[i-1]-b(i-1)*ret[i-2]
    return ret
if __name__=="__main__":
    import matplotlib.pyplot as plt
    import numpy as np
    import sys
    x=np.linspace(-1,1,100)
    legend=give_legend(x,10)
    for i in range(2,7):
        plt.plot(legend[i])
    
    plt.show()
    

