import numpy as np 
#import matplotlib.pyplot as plt
import sys
import random
import json


def load_raw_data_mnist(file_):
    t_train=[]
    t_test=[]
    labels=[]
    for line in open(file_,"r",encoding="utf-8"):
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

def load_raw_data_sst2(file_):
    t_train=[]
   
    labels=[]
    for line in open(file_,"r",encoding="utf-8"):
        feature,label=line.strip().split("\t")
        feature=[float(x)for x in feature.split("<=>")]
        #exit()
        label=int(label)
        if label not in labels:
            labels.append(label)
        t_train.append([feature,label,feature])

    random.shuffle(t_train)
    return t_train,len(t_train[0][0]),len(labels)

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
    def __init__(self, data_info="mnist"):
        if data_info == "mnist":
            self.file_ = "minist_auto_encoder"
            self.train, self.test, self.input, self.output = load_raw_data_mnist(self.file_)
        if data_info == "sst2":
            self.file_train = "xy_text_cnn_train"
            self.file_test = "xy_text_cnn_test"
            self.train,self.input, self.output = load_raw_data_sst2(self.file_train)
            self.test,self.input, self.output = load_raw_data_sst2(self.file_test)
        #self.train,self.test,self.input,self.output=load_raw_data(self.file_)
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
   
    D=give_batch(data_info="sst2")
    print(D.input)
    print(D.output)
    exit()
    for _ in range(100):
        feature,label=D.do(2000)
        print(D.ith)
    print(feature)
    print(label)
   