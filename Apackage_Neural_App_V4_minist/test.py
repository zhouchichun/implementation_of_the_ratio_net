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
        self.clip=train_config["clip"]
        self.learning_rate=train_config["LEARNING_RATE"]
        self.batch_size = train_config["BATCHSIZE"]
        self.max_iter = train_config["MAX_ITER"]
        self.epoch_save=train_config["EPOCH_SAVE"]
        self.step_each_iter=train_config['STEP_EACH_ITER']
        self.step_show=train_config['STEP_SHOW']
        self.global_steps = tf.Variable(0, trainable=False)  
        self.stru_config=stru_config
        self.decay=train_config["decay"]#False
        self.test_line=train_config["test_line"]#        
        self.bound_weight=train_config["bound_weight"]
        self.is_plot=train_config["is_plot"]
        
###############################
#P.para
##############################
        self.sess=tf.Session()
        print("openning sess")
        self.loggin.info("openning sess")
        self.build_net()
        print("building net")
        self.loggin.info("building net")
        self.build_opt()
        print("building opt")
        self.loggin.info("building opt")
        self.saver=tf.train.Saver(max_to_keep=3)
        self.initialize()
        print("net initializing")
        self.loggin.info("net initializing")
        self.D=give_batch([P.a,P.b])
    def build_net(self):
        if self.net_name=="MLP":
            self.y = N.MLP(self.stru_config)
        elif self.net_name=="RBF":
            self.y = N.RBF(self.stru_config)
        else:
            self.y = N.Poly(self.stru_config)
        self.loss= (P.b-P.a)*tf.reduce_mean(tf.sqrt((1.0+self.y.d_values[1]**2)\
                                  *(2*10.0*self.y.value)**(-1)))\
                                  +self.bound_weight*tf.reduce_mean(tf.sqrt((tf.square(self.y.value_bound_r - P.B)\
                                  +tf.square(self.y.value_bound - P.A))))
        if self.test_line:
             print("正在做直线测试")
             grad=2.0/3.141592653
             inter=0.0
             self.y.test_line(grad,inter)
             self.loss_line= (P.b-P.a)*tf.reduce_mean((1.0+self.y.d_line[0]**2)**0.5\
                                          *(2*10.0*self.y.line)**(-0.5))
             self.sess.run(tf.global_variables_initializer())
             xx=[[float(x)/self.batch_size] for x in range(1,int(3.141592653*self.batch_size))]
             
             loss_line,line,gr=self.sess.run([self.loss_line,self.y.line,self.y.d_line[0]],
                                             feed_dict={self.y.input:xx})
             print("直线的泛函值为 %s"%loss_line)
             ###########################################
             xx_rand=np.random.uniform(P.a, P.b, len(xx))
             xx_rand=np.reshape(xx_rand,[len(xx),1])
             loss_line,line,gr=self.sess.run([self.loss_line,self.y.line,self.y.d_line[0]],
                                             feed_dict={self.y.input:xx_rand})
             print("蒙特卡洛直线的泛函值为 %s"%loss_line)
             exit()

    def build_opt(self):
        if self.decay:
            self.learning_rate_d = tf.train.exponential_decay(self.learning_rate,
                                           global_step=self.global_steps,
                                           decay_steps=self.step_each_iter,decay_rate=0.95)
        else:
            self.learning_rate_d =self.learning_rate
        
        if self.clip:
            self.optimizer = tf.train.AdamOptimizer(self.learning_rate_d) 
            self.gradients = self.optimizer.compute_gradients(self.loss)
            self.capped_gradients = [(tf.clip_by_value(grad, -1*self.clip, self.clip), var) for grad, var in self.gradients if grad is not None]
            self.opt = self.optimizer.apply_gradients(self.capped_gradients,self.global_steps)
        else:
            self.opt=tf.train.AdamOptimizer(learning_rate=self.learning_rate_d).minimize(self.loss,global_step=self.global_steps)

    def initialize(self):
        ckpt=tf.train.latest_checkpoint(self.save_path)
        if ckpt!=None:
            self.saver.restore(self.sess,ckpt)
            print("init from ckpt ")
            self.loggin.info("init from ckpt ")
        else:
            self.sess.run(tf.global_variables_initializer())
    def plot(self,value,real):
        value=[-x for x in value]
        real=[-x for x in real]
        plt.plot(value)
        plt.plot(real)
        
        plt.legend(["from %s"%self.net_name,"exact solution"])
        plt.show()
    def train(self):
        st=time.time()
        for epoch in range(self.max_iter):
            print("train epoch %s of total %s epoches"%(epoch,self.max_iter))
            self.loggin.info("train epoch %s of total %s epoches"%(epoch,self.max_iter))
            for step in range(self.step_each_iter):
                intx=self.D.inter(self.batch_size)
                boundx=self.D.bound(self.batch_size)             
                boundx_r=self.D.bound_r(self.batch_size)
               
                if self.decay:
                    loss,_,gs,lrd=self.sess.run([self.loss,self.opt,self.global_steps,self.learning_rate_d],\
                                            feed_dict={self.y.input:intx, self.y.bound:boundx,\
                                            self.y.bound_r:boundx_r,})
                    if np.isnan(loss):
                        print("梯度爆了,在%s步的时候，请调整参数初始化或者调小学习步长"%step)
                        self.loggin.error("梯度爆了,在%s步的时候，请调整参数初始化或者调小学习步长"%step)
                        exit()
                else:
                    loss,_,gs=self.sess.run([self.loss,self.opt,self.global_steps],\
                                            feed_dict={self.y.input:intx, self.y.bound:boundx,\
                                            self.y.bound_r:boundx_r,})
                     
                    lrd=self.learning_rate_d
                    if np.isnan(loss):
                        print("梯度爆了,在%s步的时候，请调整参数初始化或者调小学习步长"%step)
                        self.loggin.error("梯度爆了,在%s步的时候"%step)
                        exit()
                if (step+1)%self.step_show==0:
                    print("loss %s, in epoch %s, in step %s, in global step %s, learning rate is %s, taks %s seconds"%(loss,epoch,step,gs,lrd,time.time()-st))
                    self.loggin.error("loss %s, in epoch %s, in step %s, in global step %s, learning rate is %s, taks %s seconds"%(loss,epoch,step,gs,lrd,time.time()-st))
                    st=time.time()
        
            if (epoch+1)%self.epoch_save==0:
                self.saver.save(self.sess, self.save_path+"/check.ckpt")
                r,phi=[1.0,3.141592653]
                num_point=4000.0
                the=np.linspace(P.a, P.b, int(num_point))
                xx=r*(the-np.sin(the))
                real=r*(1-np.cos(the))
                int_x=np.reshape(the,(len(the),1))
                boundx=self.D.bound(len(the))             
                boundx_r=self.D.bound_r(len(the))
                print("散点一共有%s 个"%len(int_x))
               
                if self.decay:
                    value,lrd,loss=self.sess.run([self.y.value,self.learning_rate_d,self.loss], \
                                            feed_dict={self.y.input:int_x,self.y.bound:boundx,\
                                            self.y.bound_r:boundx_r,})
                else:
                    value,loss=self.sess.run([self.y.value,self.loss], \
                                            feed_dict={self.y.input:int_x,self.y.bound:boundx,\
                                            self.y.bound_r:boundx_r,})
                    lrd=self.learning_rate_d
                if self.is_plot:
                    self.plot(value,real)
                print("Model saved in path: %s in epoch %s. learning_rate is %s" % (self.save_path,epoch,lrd))
                self.loggin.info("Model saved in path: %s in epoch %s. learning_rate is %s" % (self.save_path,epoch,lrd))
    
    def test(self):
        st=time.time()
        num_point=1000
        the=np.linspace(P.a, P.b, int(num_point))
        real=the**3+the**2+the
        dreal=3*the**2+2*the+1
        ddreal=6*the+2
        int_x=np.reshape(the,(len(the),1))
        value,dvalue,ddvalue=self.sess.run([self.y.value, self.y.d_values[1],self.y.d_values[2]],feed_dict={self.y.input:int_x})
        if self.is_plot:
            self.plot(ddvalue,ddreal)
          

if __name__ == '__main__':
    logger = logging.getLogger(C.Poly_config["which"])
    logger.setLevel(level = logging.INFO)
    handler = logging.FileHandler(C.MLP_config["which"]+"_log.txt")
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    main_net=the_net(C.train_config,C.Poly_config,logger)
    main_net.test()
