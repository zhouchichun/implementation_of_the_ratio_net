import numpy as np 
#import matplotlib.pyplot as plt
import sys
import random


def load_raw_data(file_):
    t_train=[]
    t_test=[]
    labels=[]
    for line in open("minist_auto_encoder","r",encoding="utf-8"):
        path,data=line.strip().split("\t")
        feature=[float(x)for x in data.split("<=>")]
        path=path.split("/")[-1]
        which,label,im=path.split("_")
        label=int(label)
        if label not in labels:
            labels.append(label)
        if which=="train":
            t_train.append([feature,label,path])
        else:
            t_test.append([feature,label,path])
    random.shuffle(t_train)
    random.shuffle(t_test)
    print("ok")
    return t_train,t_test,len(t_train[0][0]),len(labels)
def blank(num):
    ret=[]
    for _ in range(num):
        ret.append(0)
    return ret
def to_one_hot(label,label_length):
    ret=blank(label_length)
    ret[label]=1
    return ret
    

class give_batch():
    def __init__(self):
        self.file_ = "minist_auto_encoder"
        self.train,self.test,self.input,self.output=load_raw_data(self.file_)
        self.total_train=len(self.train)
        self.ith=0
        
    def do(self, batchsize):
        if (self.ith+1)*batchsize>self.total_train:
            random.shuffle(self.train)
            self.ith=0
            return self.do(batchsize)
        else:
            ret =self.train[self.ith*batchsize:(self.ith+1)*batchsize]
            feature=[x[0] for x in ret]
            label=[to_one_hot(x[1],self.output) for x in ret]
            self.ith +=1
            return feature,label
    def do_test_batch(self,batchsize):
        ret =self.test[batchsize:(2)*batchsize]
        feature=[x[0] for x in ret]
        label=[to_one_hot(x[1],self.output) for x in ret]
        return feature,label
    def do_test_all(self):
        feature=[x[0] for x in self.test]
        label=[to_one_hot(x[1],self.output) for x in self.test]
        return feature,label
   

if __name__ == '__main__':
   
    D=give_batch()
    print(D.input)
    print(D.output)
    for _ in range(100):
        feature,label=D.do(2000)
        print(D.ith)
    print(feature)
    print(label)
   