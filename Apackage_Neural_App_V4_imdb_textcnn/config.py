
RBF_config={
        'which':'RBF',
        'var_name':'real',
        "hidden_nodes":32,#RBF node
        "struc":16,
        "ini_name":"uniform",#"tru_norm","xavier","const","scal","uniform",orth
        }
        
MLP_config={
        'which':'MLP',
        'struc':[[128,'relu'],[128,'relu']],#'tanh','relu'
        'var_name':'real',
        "ini_name":"uniform",#"tru_norm","xavier","const","scal","uniform",orth
        }

Pade_config={
        'which':'Pade',#Pade,MLP
        'struc':[[32]],#'tanh','relu'
        'var_name':'real',
        "ini_name":"uniform",#"tru_norm","xavier","const","scal","uniform",orth
        }
train_config={
       
        'CKPT':'ckpt',
        "BATCHSIZE":512,
        "MAX_ITER":9,
        'STEP_EACH_ITER':500,
        'STEP_SHOW_train':30,
        'STEP_SHOW_test':50,
        'EPOCH_SAVE':10,
        "LEARNING_RATE":0.0001,
        "bound_weight":1,
        "step_unbound":5,
        "decay":False,
        "test_line":False,
        "is_plot":True
}
