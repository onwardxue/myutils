# -*- coding:utf-8 -*-
# @Time : 2022/7/21 9:34 下午
# @Author : Bin Bin Xue
# @File : main
# @Project : myutils

'''
    数据集：default of credit_card_clients
    链接：https://www.kaggle.com/datasets/uciml/default-of-credit-card-clients-dataset
    25个字段
    离散特征：ID、SEX、EDUCATION、MARRIAGE、
    连续特征：LIMIT_BAL、PAY_0-6、BILL_AMT1-6、PAY_AMT1-6
    离散转连续：AGE
    标签:default.payment.next.month
'''
from sklearn import ensemble
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

import preprocess_utils as dp
import model_utils as mdu
import PCA_utils as pu
import Explaining_utils as eu


path = '../data_set/credit_card_clients/default of credit card clients.csv'
# 0_数据预处理
def data_pp():
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
    return X_train,y_train,X_test,y_test

# 1_对path数据集进行初步探索、处理和分析
def data_experiment():
    X_train,y_train,X_test,y_test = data_pp()
    X = pd.concat([X_train, X_test])
    y = pd.concat([y_train, y_test])

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
    dp.plotMatrix(rf2, X_test, y_test)
    dp.plotLearning(rf2, X, y)

# 2_绘制数据集中某个特征的分布直方图
def feature_histogram(ft):
    # 获取数据
    df = dp.getData(path)
    # 绘制某个特征的分布
    dp.histogram(df,ft)

# 3_使用九个分类模型进行数据预测
def model_test():
    # 数据预处理
    X_train,y_train,X_test,y_test = data_pp()

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

# 4_pca测试
def pca_experiment():
    # 获取数据
    df = dp.getData(path)
    df = dp.del_feather(df, 'ID')
    # 划分出标签列
    df, X, y = dp.extract_label(df, 'default payment next month')
    pu.pcaTest(X)

# 5_postprocess_utils测试


# 6_Explaining_utils测试
def ExplainingTest():
    # 测试树模型（使用的是随机森林，也可以用其他的实验）的解释方法
    X_train,y_train,X_test,y_test = data_pp()
    # rf = RandomForestClassifier(random_state=42)
    # rf.fit(X_train, y_train.values.ravel())
    # score = rf.score(X_test, y_test)
    # eu.treeExplain(rf,X_train)
    # 替代解释（树模型替svm解释结果，输出特征重要性。也可以改成通用方法，但现在普遍用shap值作解释）
    # eu.replaceModel(X_train,X_test,y_train,y_test)
    # shap值测试
    rf = RandomForestClassifier(
        **{
            'min_samples_leaf':0.1,
            'n_estimators':200,
            'random_state':42,
        }
    )
    eu.shapTest(rf,X_train,y_train,X_test)

    

# 主要方法
def main():
    # 1_对数据的初步探索，简单处理和分析
    # data_experiment()
    # 2_绘制某特征的直方图
    # feature_histogram('AGE')
    # 3_九个分类模型测试(无交叉检验)
    # model_test()
    # 4_pca降维测试
    # pca_experiment()
    # 5_
    # 6_Explaining_utils测试
    ExplainingTest()


if __name__ == '__main__':
    main()
