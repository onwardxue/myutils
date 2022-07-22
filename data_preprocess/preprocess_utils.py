# -*- coding:utf-8 -*-
# @Time : 2022/7/21 9:41 下午
# @Author : Bin Bin Xue
# @File : preprocess_utils
# @Project : myutils

'''
    自定义数据预处理：
        1.读取文件:csv、feather、pickle

'''
import random

import pandas as pd

from sklearn import (
    ensemble,
    preprocessing,
    tree,
    model_selection,
    impute,
)

# 导入不同的模型库
# 准备交叉验证
from sklearn import model_selection
# 1.基础模型
from sklearn.dummy import DummyClassifier
# 2.线性回归
from sklearn.linear_model import (LogisticRegression, )
# 3.决策树
from sklearn.tree import DecisionTreeClassifier
# 4.K近邻
from sklearn.neighbors import KNeighborsClassifier
# 5.朴素贝叶斯
from sklearn.naive_bayes import GaussianNB
# 6.支持向量机
from sklearn.svm import SVC
# 7.随机森林
from sklearn.ensemble import RandomForestClassifier
# 8.xgboost
import xgboost


# 9.用stacking整合不同分类器
# from mlxtend.classifier import StackingClassifier


# 1_读取三种类型的数据
def getData(path, type='csv'):
    df = []
    if type == 'csv':
        df = pd.read_csv(path)
    elif type == 'feather':
        df = pd.read_feather(path)
    elif type == 'pickle':
        df = pd.read_pickle
    else:
        print('输入错误，请重新输入')
    return df


# 2_查看整体数据信息：特征格式、行数和列数、整体信息、各列缺失值数量
def dataInf(df):
    # 显示特征格式
    print('----------显示特征格式----------')
    print(df.dtypes)
    # 显示行数和列数
    print('----------显示行数和列数----------')
    print(df.shape)
    # 显示整体信息
    print('----------显示整体信息----------')
    print(df.describe())
    # 显示各列缺失值数量
    print('----------显示各特征缺失值数量----------')
    print(df.isnull().sum())


# 3_删除特征
def del_feather(df, feather):
    df = df.drop(columns=feather)
    return df


# 4_类型转换
def type_change(df):
    # 类别变量转为数值数值型(one-hot)
    df = pd.get_dummies(df, drop_first=True)
    return df


# 5_提出标签特征所在列
def extract_label(df, label):
    y = df[[label]]
    X = df.drop(columns=label)
    return df, X, y


# 6_划分数据集为训练集和测试集
def train_test(X, y,test_size):
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=test_size, random_state=42)
    return X_train, X_test, y_train, y_test


# 7_缺失值填充（数据集，特征，填充方法）
def deal_none(df, feather, fill_method='med'):
    num_cols = [
        feather,
    ]
    # 拟合法填充
    if fill_method == 'fit':
        imputer = impute.IterativeImputer()
        imputed = imputer.fit_transform(df[num_cols])
        df.loc[:, num_cols] = imputed
    # 中位数法填充
    elif fill_method == 'med':
        meds = df.median()
        df = df.fillna(meds)
    return df


# 8_数值数据归一化（划到0-1之间）
def dataRegular(df, cols):
    sca = preprocessing.StandardScaler()
    df = sca.fit_transform(df)
    df = pd.DataFrame(df, columns=cols)
    return df


# 9_使用八种模型对数据集进行测试，输出AUC（十折交叉验证）
def multiModel(X_train, X_test, y_train, y_test):
    # 合并数据集
    X = pd.concat([X_train, X_test])
    y = pd.concat([y_train, y_test])

    # 逐个模型进行训练
    for model in [
        DummyClassifier,
        LogisticRegression,
        DecisionTreeClassifier,
        KNeighborsClassifier,
        GaussianNB,
        SVC,
        RandomForestClassifier,
        xgboost.XGBClassifier,
    ]:
        cls = model()
        # 设置交叉检测次数和随机种子
        kfold = model_selection.KFold(
            n_splits=10, shuffle=True, random_state=42
        )
        # 使用指定的模型进行交叉检验
        s = model_selection.cross_val_score(cls, X, y.values.ravel(), scoring='roc_auc', cv=kfold)

        # 输出AUC（取10次的均值）和标准差
        print(
            f"{model.__name__:30}     AUC:  "
            f"{s.mean():.3f} STD:  {s.std():.2f}"
        )

