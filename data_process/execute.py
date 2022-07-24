# -*- coding:utf-8 -*-
# @Time : 2022/7/21 9:34 下午
# @Author : Bin Bin Xue
# @File : main
# @Project : myutils

'''
    数据集：default of credit card clients
    链接：https://www.kaggle.com/datasets/uciml/default-of-credit-card-clients-dataset
    25个字段
    离散特征：ID、SEX、EDUCATION、MARRIAGE、
    连续特征：LIMIT_BAL、PAY_0-6、BILL_AMT1-6、PAY_AMT1-6
    离散转连续：AGE
    标签:default.payment.next.month
'''
from sklearn import ensemble

import preprocess_utils as dp
import model_utils as mdu
import pandas as pd
import numpy as np

path = '../default of credit card clients.csv'


def main():
    # 获取数据
    df = dp.getData(path)
    # 查看数据信息
    dp.dataInf(df)
    # 删除特定特征
    df = dp.del_feather(df, 'ID')
    # 划分出标签列
    df, X, y = dp.extract_label(df, 'default payment next month')
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = dp.train_test(X, y, 0.3)
    # print(X_train.columns.values)
    # 数据集归一化
    X_train = dp.dataRegular(X_train, X_train.columns.values)
    X_test = dp.dataRegular(X_test, X_test.columns.values)
    # 多模型预测作为基线
    dp.multiModel(X_train, X_test, y_train, y_test)
    # 初始化随机森林模型
    rf = ensemble.RandomForestClassifier(
        n_estimators=100, random_state=42
    )
    # 随机森林模型评估，输出特征重要性
    rf = dp.modelAccess(rf, X_train, X_test, y_train, y_test)
    # 保存模型
    # dp.saveModel(rf,True)
    # 读取模型，并预测
    rf2 = dp.importModel(True, file='model.pickle')
    dp.model_predict(rf2, X_test, y_test)

    # other：网格搜索找最佳参数，优化模型
    dp.modelOpt(X_train, X_test, y_train, y_test)
    # other：绘制图表
    dp.plotMatrix(rf2,X_test,y_test)
    dp.plotLearning(rf2,X,y)

def test():
    # 获取数据
    df = dp.getData(path)
    # 绘制某个特征的分布
    dp.histogram(df,'AGE')

def test2():
    # 获取数据
    df = dp.getData(path)
    df = dp.del_feather(df, 'ID')
    # 划分出标签列
    df, X, y = dp.extract_label(df, 'default payment next month')
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = dp.train_test(X, y, 0.3)
    # print(X_train.columns.values)
    # 数据集归一化
    X_train = dp.dataRegular(X_train, X_train.columns.values)
    X_test = dp.dataRegular(X_test, X_test.columns.values)

    # 模型测试
    mdu.logisticRegressModel(X_train,y_train,X_test,y_test,X_train)
    mdu.GaussianNBModel(X_train,y_train,X_test,y_test,X_train)
    mdu.SVMModel(X_train,y_train,X_test,y_test,X_train)
    mdu.knnModel(X_train,y_train,X_test,y_test,X_train)
    mdu.DTModel(X_train, y_train, X_test, y_test, X_train)
    mdu.RFModel(X_train,y_train,X_test,y_test,X_train)
    mdu.XgbModel(X_train,y_train,X_test,y_test,X_train)
    mdu.lgbModel(X_train,y_train,X_test,y_test,X_train)
    mdu.catboostModel(X_train,y_train,X_test,y_test,X_train)
    mdu.TPOTModel(X_train,y_train,X_test,y_test,X_train)

if __name__ == '__main__':
    # main()
    # test()
    test2()