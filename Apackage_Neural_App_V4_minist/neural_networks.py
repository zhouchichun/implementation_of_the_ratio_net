# Author Zhouchichun
# 59338158@qq.com, zhouchichun@tju.edu.cn, zhouchichun@dali.edu.cn
"""
    构建一个网络,一维输入,该网络可以是MLP,RBF,Poly或者Four
    给出值，给出N阶导数，如，一阶，二阶，三阶导数等
    方法如下：
        构建
        phi = MLP(config)
        phi = RBF(config)
        phi = POLY(config)
        phi = FOUR(config)
        自变量
        feed_dict={phi.input:
                  }
        函数值
        phi.value      
        导数值
        phi.d_values[1],phi.d_values[2],phi.d_values[3]
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf 
import Legend
import para_fun as P
def get_initializer(ini_name):
    if ini_name=="tru_norm":
        weight_initialization =  tf.truncated_normal_initializer(stddev=0.1)
    elif ini_name=="xavier":
        weight_initialization =  tf.contrib.layers.xavier_initializer()
    elif ini_name=="const":
        weight_initialization = tf.constant_initializer(0.1)
    elif ini_name=="uniform":
        weight_initialization =tf.random_uniform_initializer()
    elif ini_name=="scal":
        weight_initialization = tf.variance_scaling_initializer()
    elif ini_name=="orth":
        weight_initialization = tf.orthogonal_initializer()
    else:
        print("初始化方式错误，请检查config.py文件，初始函数从['tru_norm',\
        'xavier','const','uniform','scal','orth']")
        exit()
    return weight_initialization

def get_activate(act_name):
    if act_name=="sigmoid":
        return tf.nn.sigmoid
    elif act_name=="tanh":
        return tf.nn.tanh
    elif act_name=="relu":
        return tf.nn.relu
    else:
        print("激活函数配置错误，请检查config.py文件，激活函数从['tanh','relu','sigmoid']")
        exit()
class RBF: 
    def __init__(self,config,n_input,n_output):
        self.n_input=n_input
        self.hidden_nodes=config["hidden_nodes"]
        self.var_name = config['var_name']
        self.ini_name =config["ini_name"]
        self.weight_initialization=get_initializer(self.ini_name)
        self.n_output = n_output
        #######
        print('建立RBF网络，参数如下')
        for ke,va in config.items():
            print(ke,va)
            print('---------------------------------------------')
        ######
        self.construct_input()
        self.build_value()


        
    def construct_input(self):
        print("建立placeholder")
        self.input=tf.placeholder(tf.float64, [None, self.n_input])   
    
    def build_value(self):
        print("建立RBF网络")
        self.distance=[]
        
        self.delta = tf.get_variable(self.var_name+"_delta",
                                     shape      = [self.hidden_nodes],
                                     initializer= self.weight_initialization,
                                     dtype      = tf.float64)
        self.delta_2=self.delta**2
        for i in range(self.hidden_nodes):
            this_center = tf.get_variable(self.var_name + 'center_' + str(i),
                                     shape      = [self.n_input],
                                     initializer= self.weight_initialization,
                                     dtype      = tf.float64)
            this_dist=tf.reshape(tf.reduce_sum((self.input - this_center)**2,axis=1),[-1,1])
            self.distance.append(this_dist)
           
        self.distance_ca=tf.concat(self.distance,axis=1)
        self.distance_ca=tf.reshape(self.distance_ca,[-1,self.hidden_nodes])
        self.out_hidden=tf.exp(-1.0*(self.distance_ca/self.delta_2))
       
        self.w_h2o=tf.get_variable(self.var_name + 'w_h2o',
                                   shape       = [self.hidden_nodes,self.n_output],
                                   initializer = self.weight_initialization,
                                   dtype       = tf.float64)
        self.bias=tf.get_variable(self.var_name+"_bias",
                                shape=[self.n_output],
                                initializer=self.weight_initialization,
                                dtype=tf.float64)
        self.value=tf.matmul(self.out_hidden,self.w_h2o)+self.bias
class MLP: 
    def __init__(self,config,n_input,n_output):
        self.struc =  config['struc']
        self.var_name = config['var_name']
        self.ini_name =config["ini_name"]
        self.weight_initialization=get_initializer(self.ini_name)
        self.n_output = n_output
        self.n_input=n_input
        ######
        print('建立MLP网络，参数如下')
        for ke,va in config.items():
            print(ke,va)
            print('---------------------------------------------')
        ######
        self.construct_input()
        self.build_value()
#############搭建MLP指定层数，指定每一曾的神经元个数
    def construct_input(self):
        print("建立placeholder")
        self.input=tf.placeholder(tf.float64, [None, self.n_input])   
    def build_value(self):
        print("建立网络结构")
        for i,stru in enumerate(self.struc):
            this_num,this_act=stru
            activate=get_activate(this_act)
            if i == 0:
                w = tf.get_variable(self.var_name + 'weight_' + str(0), 
                                    shape=[self.n_input, this_num], 
                                    initializer=self.weight_initialization, 
                                    dtype=tf.float64)
                b = tf.get_variable(self.var_name + 'bias_' + str(i), 
                                    shape=[this_num], 
                                    initializer=self.weight_initialization, 
                                    dtype=tf.float64)
                self.layer = activate(tf.add(tf.matmul(self.input, w), b))
               
            else:
                w = tf.get_variable(self.var_name + 'weight_' + str(i), 
                                    shape=[self.struc[i-1][0], this_num], 
                                    initializer=self.weight_initialization, 
                                    dtype=tf.float64)
                b = tf.get_variable(self.var_name + 'bias_' + str(i), 
                                    shape=[this_num], 
                                    initializer=self.weight_initialization, 
                                    dtype=tf.float64)
                self.layer = activate(tf.add(tf.matmul(self.layer, w), b))
                
        w =  tf.get_variable(self.var_name+'weight_' + str(len(self.struc)), 
                            shape=[self.struc[-1][0], self.n_output], 
                            initializer=self.weight_initialization, 
                            dtype=tf.float64)
        b =tf.get_variable(self.var_name+'bias_' + str(len(self.struc)), 
                            shape=[self.n_output], 
                            initializer=self.weight_initialization, 
                            dtype=tf.float64)
        self.value=tf.matmul(self.layer, w) + b
##############
class Pade: 
    def __init__(self,config,n_input,n_output):
        self.var_name = config['var_name']
        self.ini_name =config["ini_name"]
        self.struc =  config['struc']
        self.weight_initialization=get_initializer(self.ini_name)
        self.n_output = n_output
        self.n_input=n_input
        ######
        print('建立MLP网络，参数如下')
        for ke,va in config.items():
            print(ke,va)
            print('---------------------------------------------')
        ######
        self.construct_input()
        self.build_value()

    def construct_input(self):
        print("建立placeholder")
        self.input=tf.placeholder(tf.float64, [None, self.n_input])   
    def build_value(self):
        print("建立网络结构")
        for i,stru in enumerate(self.struc):
            this_num=stru[0]
            if i == 0:
                w1 = tf.get_variable(self.var_name + 'weight_1' + str(0), 
                                    shape=[self.n_input, this_num], 
                                    initializer=self.weight_initialization, 
                                    dtype=tf.float64)
                b1 = tf.get_variable(self.var_name + 'bias_1' + str(i), 
                                    shape=[this_num], 
                                    initializer=self.weight_initialization, 
                                    dtype=tf.float64)
                self.layer1 = tf.add(tf.matmul(self.input, w1), b1)#activate(tf.add(tf.matmul(self.input, w), b))
                w2 = tf.get_variable(self.var_name + 'weight_2' + str(0), 
                                    shape=[self.n_input, this_num], 
                                    initializer=self.weight_initialization, 
                                    dtype=tf.float64)
                b2 = tf.get_variable(self.var_name + 'bias_2' + str(i), 
                                    shape=[this_num], 
                                    initializer=self.weight_initialization, 
                                    dtype=tf.float64)
                self.layer2 = tf.add(tf.matmul(self.input, w2), b2)#activate(tf.add(tf.matmul(self.input, w), b))
                w3 = tf.get_variable(self.var_name + 'weight_3' + str(0), 
                                    shape=[self.n_input, this_num], 
                                    initializer=self.weight_initialization, 
                                    dtype=tf.float64)
                b3 = tf.get_variable(self.var_name + 'bias_3' + str(i), 
                                    shape=[this_num], 
                                    initializer=self.weight_initialization, 
                                    dtype=tf.float64)
                self.layer3 = tf.add(tf.matmul(self.input, w3), b3)#activate(tf.add(tf.matmul(self.input, w), b))
                w4 = tf.get_variable(self.var_name + 'weight_4' + str(0), 
                                    shape=[self.n_input, this_num], 
                                    initializer=self.weight_initialization, 
                                    dtype=tf.float64)
                b4 = tf.get_variable(self.var_name + 'bias_4' + str(i), 
                                    shape=[this_num], 
                                    initializer=self.weight_initialization, 
                                    dtype=tf.float64)
                self.layer4 = tf.add(tf.matmul(self.input, w4), b4)
                self.layer=self.layer1*self.layer2/(self.layer3*self.layer4)
            else:
                w1 = tf.get_variable(self.var_name + 'weight_1' + str(i), 
                                    shape=[self.struc[i-1][0], this_num], 
                                    initializer=self.weight_initialization, 
                                    dtype=tf.float64)
                b1 = tf.get_variable(self.var_name + 'bias_1' + str(i), 
                                    shape=[this_num], 
                                    initializer=self.weight_initialization, 
                                    dtype=tf.float64)
                self.layer1 = tf.add(tf.matmul(self.layer, w1), b1)#activate(tf.add(tf.matmul(self.layer, w), b))
                w2 = tf.get_variable(self.var_name + 'weight_2' + str(i), 
                                    shape=[self.struc[i-1][0], this_num], 
                                    initializer=self.weight_initialization, 
                                    dtype=tf.float64)
                b2 = tf.get_variable(self.var_name + 'bias_2' + str(i), 
                                    shape=[this_num], 
                                    initializer=self.weight_initialization, 
                                    dtype=tf.float64)
                self.layer2 = tf.add(tf.matmul(self.layer, w2), b2)
                w3 = tf.get_variable(self.var_name + 'weight_3' + str(i), 
                                    shape=[self.struc[i-1][0], this_num], 
                                    initializer=self.weight_initialization, 
                                    dtype=tf.float64)
                b3 = tf.get_variable(self.var_name + 'bias_3' + str(i), 
                                    shape=[this_num], 
                                    initializer=self.weight_initialization, 
                                    dtype=tf.float64)
                self.layer3 = tf.add(tf.matmul(self.layer, w3), b3)#activate(tf.add(tf.matmul(self.layer, w), b))
                w4 = tf.get_variable(self.var_name + 'weight_4' + str(i), 
                                    shape=[self.struc[i-1][0], this_num], 
                                    initializer=self.weight_initialization, 
                                    dtype=tf.float64)
                b4 = tf.get_variable(self.var_name + 'bias_4' + str(i), 
                                    shape=[this_num], 
                                    initializer=self.weight_initialization, 
                                    dtype=tf.float64)
                self.layer4 = tf.add(tf.matmul(self.layer, w4), b4)
                
                self.layer=self.layer1*self.layer2/(self.layer3*self.layer4)
        w =  tf.get_variable(self.var_name+'weight_' + str(len(self.struc)), 
                            shape=[self.struc[-1][0], self.n_output], 
                            initializer=self.weight_initialization, 
                            dtype=tf.float64)
        b =tf.get_variable(self.var_name+'bias_' + str(len(self.struc)), 
                            shape=[self.n_output], 
                            initializer=self.weight_initialization, 
                            dtype=tf.float64)
        self.value=tf.matmul(self.layer, w) + b
        