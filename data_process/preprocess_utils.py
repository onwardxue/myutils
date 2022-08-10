# -*- coding:utf-8 -*-
# @Time : 2022/7/21 9:41 下午
# @Author : Bin Bin Xue
# @File : preprocess_utils
# @Project : myutils

'''
    自定义数据预处理，分为多个部分：
    a.一个完整的数据分析流程，用于对表格数据进行简要的探索
    b.模型优化
    c.

'''
import random

# 模型保存和加载库
import pickle

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn import (
    ensemble,
    preprocessing,
    tree,
    model_selection,
    feature_selection,
)

from sklearn.metrics import (
    auc,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    precision_score,

)

from sklearn.impute import (
    SimpleImputer,
)

from sklearn.experimental import enable_iterative_imputer

from yellowbrick.classifier import (
    ConfusionMatrix,
    ROCAUC,
)

from yellowbrick.model_selection import (
    LearningCurve,
)

from yellowbrick.features import RFECV

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


# a.一个完整的数据分析过程，简要的对数据进行初步的分析与处理
# 1_读取三种类型的数据（csv、feather、pickle）
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


# 4_类型转换（类别特征）
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
def train_test(X, y, test_size):
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=test_size, random_state=42)
    return X_train, X_test, y_train, y_test


# 7_缺失值填充（数据集，填充方法，指定特征）
def deal_none(df, fill_method='med',feather=None,fvalue=-1):

    # 如果没有指定要填充的特征，挑选出数据中数值型的特征进行缺失值填充
    if feather is  None:
        num_cols = df.select_dtypes(include='number').columns
    else:
        num_cols = list(feather)

    # 拟合法填充
    if fill_method == 'fit':
        im = enable_iterative_imputer.IterativeImputer()
        imputed = im.fit_transform(df[num_cols])
        df.loc[:, num_cols] = imputed
    # 插值法填充
    else:
        # 插值（默认使用均值，设置参数strategy='median'或'most_frequent使用中位数和最高频特征值，自定义常数值为'constant',fill_value=-1)
        allowed_strategies = ["mean", "median", "most_frequent", "constant"]
        if fill_method == 'constant':
            im = SimpleImputer(strategy=fill_method,fill_value=fvalue)
        else:
            im = SimpleImputer(strategy=fill_method,)
        imputed = im.fit_transform(df[num_cols])
        df.loc[:, num_cols] = imputed
    return df


# 8_数值数据归一化（数值特征划到0-1之间）
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


# 10_模型评估，重要特征
def modelAccess(model, X_train, X_test, y_train, y_test):
    # 模型训练
    model.fit(X_train, y_train.values.ravel())

    # 模型评估
    # 返回准确率
    accuracy = model.score(X_test, y_test)
    print('RF_accuracy：')
    print(accuracy)
    # 返回精确值
    pre = precision_score(y_test, model.predict(X_test))
    print('RF_precision：')
    print(pre)
    # 查看模型中的特征重要性(默认使用的是"gini",返回前10个重要特征）
    print('特征重要性排序：')
    for col, val in sorted(
            zip(
                X_train.columns,
                model.feature_importances_,
            ),
            key=lambda x: x[1],
            reverse=True,
    )[:10]:
        print(f"{col:10}{val:10.3f}")
    # 返回训练好的模型
    return model


# 11_模型持久化（保存训练好的模型，需要的时候再使用）
def saveModel(model, save=False, file='model.pickle'):
    # 保存模型
    if save == False:
        pic = pickle.dumps(model)
        print('模型保存完毕,返回变量...')
        return pic
    else:
        with open(file, 'wb') as f:
            pickle.dump(model, f)
        print('模型保存至指定文件...')


# 12_模型导入
def importModel(from_file=False, pic=None, file=None):
    # 加载模型
    if from_file == False:
        model = pickle.loads(pic)
        print('模型导入成功...')
        return model
    else:
        with open(file, 'rb') as f:
            res = pickle.load(f)
            return res


# 13_模型预测
def model_predict(model, X_test, y_test):
    # 用加载的模型预测当前的数据
    y_pred = model.predict(X_test)
    roc = roc_auc_score(y_test, y_pred)
    print('预测结果为：%f' % roc)


# b.模型优化

#  模型优化：网格搜索法调整模型训练参数
def modelOpt(X_train, X_test, y_train, y_test):
    # 初始化随机森林
    rf4 = ensemble.RandomForestClassifier()
    # 设置要调整的参数范围
    params = {
        "n_estimators": [15, 200],
        "min_samples_leaf": [1, 0.1],
        "random_state": [42],
    }
    # 使用网格搜索法设置参数
    cv = model_selection.GridSearchCV(
        rf4, params, n_jobs=-1).fit(X_train, y_train.values.ravel())

    # 输出最佳参数值
    print('最佳模型参数为：')
    print(cv.best_params_)

    rf5 = ensemble.RandomForestClassifier(
        **{
            "min_samples_leaf": 1,
            "n_estimators": 200,
            "random_state": 42
        }
    )
    rf5.fit(X_train, y_train.values.ravel())
    auc = rf5.score(X_test, y_test.values.ravel())
    print('调参后的模型预测Auc为：')
    print(auc)


# c.绘制图表和曲线
# 绘制混淆矩阵
def plotMatrix(model, X_test, y_test):
    # 取得预测标签
    y_pred = model.predict(X_test)
    # 用真实标签和预测标签得到混淆矩阵
    cfs = confusion_matrix(y_test, y_pred)
    print(cfs)

    mapping = {0: 'Negative', 1: 'Positive'}
    fig, ax = plt.subplots(figsize=(6, 6))
    cm_viz = ConfusionMatrix(
        model,
        classes=['Negative', 'Positive'],
        label_encoder=mapping,
    )
    cm_viz.score(X_test, y_test)
    cm_viz.show()
    # if save_name is not None:
    #     fig.savefig(
    #         save_name,
    #         dpi=300,
    #         bbox_inches='tight',
    #     )


# 绘制三种曲线：Roc曲线（预测性能），Learning曲线（显示训练数据的量是否足够支持模型）

def plotLearning(model, X, y):
    fig, ax = plt.subplots(figsize=(6, 4))
    cv = model_selection.StratifiedKFold(12)
    sizes = np.linspace(0.3, 1.0, 10)
    lc_viz = LearningCurve(
        model,
        cv=cv,
        train_sizes=sizes,
        scoring='f1_weighted',
        n_jobs=4,
        ax=ax,
    )
    lc_viz.fit(X, y)
    lc_viz.show()
    # fig.savefig('images/mlpr_030.png')

# d 特征分析

# 绘制直方图（查看某个特征的数值分布）
def histogram(df,feature):
    fig, ax = plt.subplots(figsize=(6, 4))
    df[feature].plot(kind='hist', ax=ax)
    plt.show()
    # fig.savefig('images/mlpr_060.png', dpi=300)

# e 类别特征的其他编码（除了one-hot，其他还有标签编码、频数编码、哈希编码、序数编码。。）

# f 查看特征重要性（特征筛选或降维）
# 1.使用rfecv算法进行特征筛选（递归消除不重要特征）
def RFECV_screen(X,y):
    fig, ax = plt.subplots(figsize=(6, 4))
    rfe = RFECV(
        ensemble.RandomForestClassifier(
            n_estimators=100
        ),
        cv=5,
    )
    rfe.fit(X,y)
    print(rfe.rfe_estimator_.ranking_)
    print(rfe.rfe_estimator_.n_features_)
    print(rfe.rfe_estimator_.support_)
    rfe.show()
    # fig.savefig('images/mlpr_083.png',dpi=300)


# 2.使用互信息进行特征筛选（递归消除不重要特征）
def multinfo_screen(X,y):
    mic = feature_selection.mutual_info_classif(
        X, y
    )
    fig, ax = plt.subplots(figsize=(10, 8))
    (
        pd.DataFrame(
            {'feature': X.columns, 'vimp': mic}
        )
            .set_index('feature')
            .plot.barh(ax=ax)
    )
    plt.show()
    # fig.savefig('images/mlpr_084.png')

# 3.使用PCA进行降维



# g 类不平衡数据处理
# 1.过采样和欠采样
#2.使用imblanced-learn库实现

