# 数据说明
- xy_text_cnn_train_1000 和 xy_text_cnn_test_1000   是IMDb数据经过text_cnn有监督训练以后，提取的特征（Full connect前的那一层），维度是384维。
  由于原始数据大于100MB，因此，我截了1000条数据作为示例。需要原始数据请联系  59338158@qq.com
- 数据格式是  0.0<=>1321.6782<=>1925.9651<=>1955.5665...\tlabel
# 运行说明
  传统网络：python main.py MLP
  新网络：python main.py Pade
  在这个 Pade 就是 ratio net
  
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
