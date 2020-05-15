import os
import glob
import matplotlib.pyplot as plt
import math
import para_fun as P
ckpt_path=glob.glob("*log.txt")
print(ckpt_path)
#exit()
def give_lst(path_):
    print(path_)
    ret=[]
    step=0
    name="_".join(path_.split("_")[:-1])
    for line in open(path_,"r",encoding="utf-8"):
        if line.find("test->")!=-1:
            #print(line)
            #exit()
            loss=float(line.split("acc")[-1].split("is ")[-1].split(",")[0].strip())
            step +=1#int(line.split("global step")[-1].split(",")[0].strip())
            ret.append([step,loss])
    return ret,name
max_len=0
t={}
for ckpt in ckpt_path:
    
    lst,name=give_lst(ckpt)
    t[name]=lst
    if len(lst) >max_len:
        max_len=len(lst) 
print(t)
legend=[]
for name, lst in t.items():    
    legend.append(name)
    #print(lst[:10])
    #exit()
    x=[xx[0]for xx in lst]
    y=[xx[1] for xx in lst]
    #for xx in lst:
     #   if xx[1] >0:
     #      y.append(math.log(xx[1]))
     #   else:
     #       y.append(-math.log(-xx[1]))
       
    plt.plot(x,y)
#plt.plot(x,[math.log(P.exact+0.5)]*len(x))
#plt.plot(x,[P.exact]*len(x))
#legend.append("exact")
plt.legend(legend)
plt.xlabel('steps')
plt.ylabel('acc_test')
plt.show()
    