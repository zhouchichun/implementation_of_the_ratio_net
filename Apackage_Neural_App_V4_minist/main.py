import tensorflow as tf 

import numpy as np 
import neural_networks as N
import matplotlib.pyplot as plt
import sys, getopt
from utils import give_batch
import config as C
import time    
import para_fun as P
import logging


class the_net():
    def __init__(self,train_config,stru_config,loggin):
        self.net_name=stru_config["which"]
        self.loggin=loggin
        self.loggin.info("now initialize the net with para:")
        for item,value in train_config.items():
            self.loggin.info(str(item))
            self.loggin.info(str(value))
            self.loggin.info("-----------------------------")
        
        self.save_path=train_config["CKPT"]+"_"+self.net_name
        
        self.learning_rate=train_config["LEARNING_RATE"]
        self.batch_size = train_config["BATCHSIZE"]
        self.max_iter = train_config["MAX_ITER"]
        #self.step_unbound=train_config["step_unbound"]
        self.epoch_save=train_config["EPOCH_SAVE"]
        self.step_each_iter=train_config['STEP_EACH_ITER']
        self.step_show_train=train_config['STEP_SHOW_train']
        self.step_show_test=train_config['STEP_SHOW_test']
        
        self.global_steps = tf.Variable(0, trainable=False)  
        self.stru_config=stru_config
       

        self.D=give_batch()
        self.sess=tf.Session()
        print("openning sess")
        self.loggin.info("openning sess")
        self.build_net()
        self.build_loss()
        print("building net")
        self.loggin.info("building net")
        self.build_opt()
        print("building opt")
        self.loggin.info("building opt")
        self.saver=tf.train.Saver(max_to_keep=3)
        self.initialize()
        print("net initializing")
        self.loggin.info("net initializing")
        self.D=give_batch()

    def build_net(self):
        self.target=tf.placeholder(tf.float64, [None, self.D.output])
        if self.net_name=="MLP":
            self.y = N.MLP(self.stru_config,self.D.input,self.D.output)
        elif self.net_name=="Pade":
            self.y = N.Pade(self.stru_config,self.D.input,self.D.output)
        elif self.net_name=="RBF":
            self.y = N.RBF(self.stru_config,self.D.input,self.D.output)
        else:
            print("网络模型错误！")
            exit()
###########################################
    def build_loss(self):
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.y.value,
                                                                           labels=self.target,
                                                                           name='xentropy_per_example')
        self.loss = tf.reduce_mean(cross_entropy, name='loss')
    def cal_acc(self,pre,real):
        right=0
        
        for p,r in zip(pre,real):
            p=list(p)
            r=list(r) 
            if p.index(max(p))==r.index(max(r)):
                right +=1
        return float(right)/len(real)
        
    def build_opt(self):
        self.learning_rate_d =self.learning_rate
        self.opt=tf.train.AdamOptimizer(learning_rate=self.learning_rate_d).minimize(self.loss,global_step=self.global_steps)

    def initialize(self):
        ckpt=tf.train.latest_checkpoint(self.save_path)
        if ckpt!=None:
            self.saver.restore(self.sess,ckpt)
            print("init from ckpt ")
            self.loggin.info("init from ckpt ")
        else:
            self.sess.run(tf.global_variables_initializer())
        
    def train(self):
        st=time.time()
        for epoch in range(self.max_iter):
            print("train epoch %s of total %s epoches"%(epoch,self.max_iter))
            self.loggin.info("train epoch %s of total %s epoches"%(epoch,self.max_iter))
            for step in range(self.step_each_iter):
                feature,label=self.D.do(self.batch_size)
                loss,_,gs,logit=self.sess.run([self.loss,self.opt,self.global_steps,self.y.value],\
                                        feed_dict={self.y.input:feature,self.target:label})
                if (step+1)%self.step_show_train==0:
                    acc=self.cal_acc(logit,label)
                    print("loss %.4f, in epoch %s, in step %s, in global step %s,\
                    acc is %s, taks %s seconds"%(loss,epoch,step,gs,acc,time.time()-st))
                    self.loggin.info("loss %s, in epoch %s, in step %s, in global step %s, acc is %s,\
                    taks %s seconds"%(loss,epoch,step,acc,acc,time.time()-st))
                    st=time.time()
                if (step+1)%self.step_show_test==0:
                    feature,label=self.D.do_test_batch(self.batch_size)
                    loss,logit=self.sess.run([self.loss,self.y.value],\
                                        feed_dict={self.y.input:feature,self.target:label})
                    acc=self.cal_acc(logit,label)
                    print("test->loss %.4f, in epoch %s, in step %s, in global step %s,\
                    acc is %s, taks %s seconds"%(loss,epoch,step,gs,acc,time.time()-st))
                    self.loggin.info("test->loss %s, in epoch %s, in step %s, in global step %s, \
                    acc is %s, taks %s seconds"%(loss,epoch,step,gs,acc,time.time()-st))
                    st=time.time()
            if (epoch+1)%self.epoch_save==0:
                self.saver.save(self.sess, self.save_path+"/check.ckpt")
                print("Model saved in path: %s in epoch %s. acc is %s, loss is %s" % (self.save_path,epoch,acc,loss))
                self.loggin.info("Model saved in path: %s in epoch %s. acc is %s, loss is %s" % (self.save_path,epoch,acc,loss))
            

if __name__ == '__main__':
    which=sys.argv[1]
    if which=="MLP":
        logger = logging.getLogger(C.MLP_config["which"])
        logger.setLevel(level = logging.INFO)
        handler = logging.FileHandler(C.MLP_config["which"]+"_"+str(C.MLP_config["struc"])+"_log.txt")
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        main_net=the_net(C.train_config,C.MLP_config,logger)
        main_net.train()
    if which=="RBF":
        logger = logging.getLogger(C.RBF_config["which"])
        logger.setLevel(level = logging.INFO)
        handler = logging.FileHandler(C.RBF_config["which"]+"_"+str(C.RBF_config["struc"])+"_log.txt")
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        main_net=the_net(C.train_config,C.RBF_config,logger)
        main_net.train()
    
    if which=="Pade":
        logger = logging.getLogger(C.Pade_config["which"])
        logger.setLevel(level = logging.INFO)
        handler = logging.FileHandler(C.Pade_config["which"]+"_"+str(C.Pade_config["struc"])+"_log.txt")
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        main_net=the_net(C.train_config,C.Pade_config,logger)
        main_net.train()
  