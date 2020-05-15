# 数据说明
- minist_auto_encoder   是minist数据经过卷积自编码提取的特征，维度是20维。
- 数据格式是   train_label_samplename/t0.0<=>1321.6782<=>1925.9651<=>1955.5665
# 运行说明
  传统网络：python main.py MLP
  新网络：python main.py Pade    
  
# 配置网络结构说明
在config.py文件中，配置 struc
```
MLP_config={
        'which':'MLP',
        'struc':[[128,'sigmoid'],[128,'sigmoid']],#'tanh','relu'
        'var_name':'real',
        "ini_name":"uniform",#"tru_norm","xavier","const","scal","uniform",orth
        }

Pade_config={
        'which':'Pade',#Pade,MLP
        'struc':[[16,'sigmoid']],#'tanh','relu'
        'var_name':'real',
        "ini_name":"uniform",#"tru_norm","xavier","const","scal","uniform",orth
        }
```