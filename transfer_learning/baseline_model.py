# -*- coding:utf-8 -*-
# @Time : 2022/7/10 3:25 下午
# @Author : Bin Bin Xue
# @File : baseline_model
# @Project : tradaboostTest

import xgboost as xgb
import pandas as pd
from sklearn.metrics import roc_auc_score

# 用train_b直接训练xgb模型，对test_b数据进行预测
# 目录
import os

p = os.path.dirname(os.getcwd())
after_path = p + '/data_set/loans/data_after/'
before_path = p + '/data_set/loans/data/'
result_path = p + '/data_set/loans/result/'
# 读取数据
# 源域
train_A = pd.read_csv(after_path + 'A_train.csv')
train_A_flag = pd.read_csv(after_path + 'A_train_flag.csv')
# 目标域
train_B = pd.read_csv(after_path + 'B_train.csv')
train_B_flag = pd.read_csv(after_path + 'B_train_flag.csv')
# test = pd.read_csv(before_path+'B_test.csv')

# 转成适合xgb内部使用的数据
dtrain_A = xgb.DMatrix(data=train_A, label=train_A_flag)
dtrain_B = xgb.DMatrix(data=train_B, label=train_B_flag)

# 1_训练源域模型
Trate = 0.25
# params = {'booster': 'gbtree', 'eta': 0.1, 'max_depth': 4, 'max_delta_step': 0, 'subsample': 0.9,
#           'colsample_bytree': 0.9, 'base_score': Trate, 'objective': 'binary:logistic', 'lambda': 5, 'alpha': 8,
#           'random_seed': 100, 'eval_metric': 'auc'}
# xgb_model = xgb.train(params, dtrain_B, num_boost_round=200, maximize = True,
#                       verbose_eval= True )
xgb_class = xgb.XGBClassifier(random_state=42, early_stopping_rounds=10)
xgb_class.fit(
    train_A,
    train_A_flag.values.ravel(),
    eval_set=[(train_A, train_A_flag.values.ravel())],
)
# xgb_model = xgb.fit(params, dtrain_A, num_boost_round=200, maximize=True,
#                     verbose_eval=True)
auc = roc_auc_score(train_B_flag, xgb_class.predict(train_B))
print('auc：', auc)
# prediction = xgb_model.predict(xgb.DMatrix(test[train_B.columns].fillna(-999)))
# test['pred'] = prediction
# test[['no','pred']].to_csv(result_path+'submission.csv', index = None)

# for x in test['pred']:
#     if(x>0.5):
#         test['label'] = 1
#     else:
#         test['label'] = 0
#
# test[['no','label']].to_csv(result_path+'rs.csv',index = None)
