# -*- coding:utf-8 -*-
# @Time : 2022/8/17 7:22 下午
# @Author : Bin Bin Xue
# @File : test_1
# @Project : myutils

'''
1.17个模型：
Feature-based SFA [7], mSDA [8], SDA [9], GFK [10], SCL [11], TCA [12], JDA [13]
Concept-based HIDC [3], TriTL [4], CD-PLSA [5], MTrick [6]
Parameter-based LWE [15]
Instance-based TrAdaBoost [14]
Deep-learning-based DAN [16], DCORAL [17], MRAN [18], DANN [19][20]

2.
'''
import os
import pandas as pd
from sklearn import preprocessing
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import numpy as np
import Toolkit.traditional.instance.TrAdaBoost as tab
from sklearn.impute import SimpleImputer     #引入skleran的数据填充方法；

p = os.path.dirname(os.getcwd())
after_path = p + '/data_set/loans/data_after/'
before_path = p + '/data_set/loans/data/'
result_path = p + '/data_set/loans/result/'
# 读取数据
# 源域
train_A = pd.read_csv(before_path + 'A_train.csv')
train_A_flag = pd.read_csv(after_path + 'A_train_flag.csv')
# 目标域
train_B = pd.read_csv(before_path + 'B_train.csv')
train_B_flag = pd.read_csv(after_path + 'B_train_flag.csv')
test_B = pd.read_csv(before_path+'B_test.csv')
# test = pd.read_csv(before_path+'B_test.csv')

# 保持数据域一致
train_B_info = train_B.describe()
useful_col = []
for col in train_B_info.columns:
    if train_B_info.loc['count', col] > train_B.shape[0] * 0.01:
        useful_col.append(col)
print(useful_col)
train_B_1 = train_B[useful_col].copy()
# train_B_1 = train_B_1.fillna(-999)
relation = train_B_1.corr()

# 2.保留train_A中和train_B一样的特征（只留下源域中与目标域存在的特征）
train_A_1 = train_A[useful_col].copy()


# 3.对线性相关特征进行处理
length = relation.shape[0]
high_corr = list()
final_cols = []
del_cols = []
for i in range(length):
    if relation.columns[i] not in del_cols:
        final_cols.append(relation.columns[i])
        for j in range(i + 1, length):
            if (relation.iloc[i, j] > 0.98) and (relation.columns[j] not in del_cols):
                del_cols.append(relation.columns[j])

train_B_1 = train_B_1[final_cols]
train_A_1 = train_A_1[final_cols]

# 4.取出标签列和去除标识列
train_B_flag = train_B_1['flag']
train_B_1.drop('no', axis=1, inplace=True)
train_B_1.drop('flag', axis=1, inplace=True)

train_A_flag = train_A_1['flag']
train_A_1.drop('no', axis=1, inplace=True)
train_A_1.drop('flag', axis=1, inplace=True)

# 缺失值处理（对决策树影响较大）
imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean')#利用均值填充数据集；
imputer2 = SimpleImputer(missing_values = np.nan, strategy = 'mean')#利用均值填充数据集；
train_A_1=pd.DataFrame(imputer.fit_transform(train_A_1))
train_B_1=pd.DataFrame(imputer.fit_transform(train_B_1))


print(train_A_1.isnull().sum())
print(train_B_1.isnull().sum())

# 归一化
sca = preprocessing.StandardScaler()
df = sca.fit_transform(train_A_1)
train_A_1 = pd.DataFrame(df)
df = sca.fit_transform(train_B_1)
train_B_1 = pd.DataFrame(df)

# 5.设置可以复现的随机种子 ?问题：这一步有什么用？
def seed_everything(seed=0):
    # random.seed(seed)
    # 获取环境变量
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)


seed_everything(456)

# 6.划分目标域train_B_1训练集为valid和test（5：5划分）
train_B_1_valid, train_B_1_test, train_B_1_valid_y, train_B_1_test_y = train_test_split(train_B_1, train_B_flag,
                                                                                        test_size=0.5)


tb= tab.TrAdaBoost()
# prediction = tb.fit_predict(train_A_1,train_B_1,train_A_flag,train_B_flag,test_B)
prediction = tb.fit_predict(train_A_1.values, train_B_1_valid.values, train_A_flag, train_B_1_valid_y.values,train_B_1_test)
print(prediction)
print('auc：',roc_auc_score(train_B_1_test_y,prediction))
