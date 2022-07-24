# -*- coding:utf-8 -*-
# @Time : 2022/7/24 3:37 下午
# @Author : Bin Bin Xue
# @File : model_utils
# @Project : myutils

'''
    描述：当数据预处理完成后，这里提供一些最常见的分类任务用的模型训练方法
        其中，逻辑回归、朴素贝叶斯、支持向量机、KNN、决策树、随机森林都是用的skl库进行实现。
            skl模型的通用方法
            fit(X,y[,sample_weight]) - 训练模型
            predict(X) - 预测类别
            predict_log_proba(X) - 预测样例属于某个类别的概率的对数值
            predict_proba(X) - 预测样例属于某个类别的概率
            score(X,y[,sample_weight]) - 获取模型的准确率
        xgb使用的是xgboost库
        lgbm使用的是lightgbm库
        catboost使用的是catboost库
        TPOT使用的是tpot库
'''

# 1_skl的逻辑回归分类器
from sklearn.linear_model import (
    LogisticRegression,
)
# 2_skl的朴素贝叶斯分类器
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB
# 3_skl的svm分类器
from sklearn.svm import SVC
# 4_skl的knn分类器
from sklearn.neighbors import KNeighborsClassifier
# 5_skl的决策树分类器
from sklearn.tree import DecisionTreeClassifier
# 6_skl的随机森林分类器
from sklearn.ensemble import RandomForestClassifier
# from rfpimp import permutation_importances
from sklearn.metrics import r2_score, roc_auc_score
# 7_xgb的xgboost分类器
import xgboost as xgb
# xgbfir是一个xgboost模型转储解析器,它根据不同的度量标准对特性和特性交互进行排序
import xgbfir
# 8_lgb的lightgbm分类器
import lightgbm as lgb
# 9_catboost的catboost分类器
import catboost as cb
# 10_TPOT分类器
from tpot import TPOTClassifier

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch

# 逆对数概率函数
from yellowbrick.model_selection import FeatureImportances


# 反对数
def inv_logit(p):
    return np.exp(p) / (1 + np.exp(p))


# 计算auc(第一个是训练集结果，第二个是测试集结果)
def auc(model, X_train, y_train, X_test, y_test):
    return (roc_auc_score(y_train, model.predict_proba(X_train)[:, 1]),
            roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]))


# 1_逻辑回归分类器
# 传入的参数是已划分好的源域训练集+标签、源域测试集+标签、目标域数据集（不含标签），要修改逻辑回归参数要在这直接改
def logisticRegressModel(X_train, y_train, X_test, y_test, target):
    # 1.模型初始化（在这里设置模型参数）
    # 12个参数-penalty,dual,C,fit_intercept,intercept_scaling,max_iter,multi_class,class_weight,solver,tol,verbose,warm_start,n_jobs

    lr = LogisticRegression(random_state=42)
    # 2.模型训练（在这里加入训练数据，训练模型）
    lr.fit(X_train, y_train.values.ravel())
    # 3.输出模型结果（预测准确度）
    score = lr.score(X_test, y_test)
    auc_value = auc(lr, X_train, y_train, X_test, y_test)
    # 4.输出预测的目标类别（预测结果，各类别概率，概率的对数值）
    p = lr.predict(target)
    pp = lr.predict_proba(target)
    plp = lr.predict_log_proba(target)
    y_score = lr.decision_function(target)
    # 5.模型训练后的属性
    # 决策函数的截距 - 判正的基础概率
    # print(lr.intercept_)
    # print(inv_logit(lr.intercept_))
    # 决策函数的系数（权重）- 几次方的系数
    coef = lr.coef_
    # 决策函数的迭代次数
    iter = lr.n_iter_
    print(auc_value)

    # 5.查看各特征的系数（跟预测结果正/负相关）
    cols = target.columns
    for col, val in sorted(
            zip(cols, lr.coef_[0]),
            key=lambda x: x[1],
            reverse=True,
    ):
        print(
            f"{col:10}{val:10.3f}{inv_logit(val):10.3f}"
        )


# 2_朴素贝叶斯分类器
def GaussianNBModel(X_train, y_train, X_test, y_test, target):
    # 初始化模型（设置模型参数）
    nb = GaussianNB(priors=None, var_smoothing=1e-9)
    nb.fit(X_train, y_train.values.ravel())
    score = nb.score(X_test, y_test)
    print('BYS result：')
    print(score)
    # 输出预测结果
    print(nb.predict(target))
    print(nb.predict_proba(target))
    print(nb.predict_log_proba(target))
    # 模型训练后的属性
    print(nb.class_prior_)
    print(nb.class_count_)
    print(nb.theta_)
    print(nb.var_)
    print(nb.epsilon_)


# 3_SVM分类器
def SVMModel(X_train, y_train, X_test, y_test, target):
    # 有14个参数，其中probability=True开启后会降低速度
    svc = SVC(random_state=42, probability=True)
    svc.fit(X_train, y_train.values.ravel())
    # 输出预测结果
    score = svc.score(X_test, y_test)
    print('SVM result：')
    print(score)
    print(svc.predict(target))
    print(svc.predict_proba(target))
    print(svc.predict_log_proba(target))


# 4_K近邻分类器
def knnModel(X_train, y_train, X_test, y_test, target):
    knc = KNeighborsClassifier()
    knc.fit(X_train, y_train.values.ravel())
    score = knc.score(X_test, y_test)
    print('KNN result：')
    print(score)
    print(knc.predict(target))
    print(knc.predict_proba(target))


# 绘制决策树
# def DTGraph(dt):
#     dot_data = StringIO()
#     tree.export_graphviz(
#         dt,
#         out_file=dot_data,
#         feature_names=X.columns,
#         class_names=['Died', 'Survived'],
#         filled=True,
#     )
#     g = pydotplus.graph_from_dot_data(
#         dot_data.getvalue()
#     )
#     g.write_png('images/mlpr_102.png')


# 5_决策树分类器
def DTModel(X_train, y_train, X_test, y_test, target):
    dt = DecisionTreeClassifier(random_state=42, max_depth=3)
    dt.fit(X_train, y_train.values.ravel())
    score = dt.score(X_test, y_test)
    print('DT result：')
    print(score)
    print(dt.predict(target))
    print(dt.predict_proba(target))
    # print(dt.predict_log_proba(X.iloc[[0]]))
    # 模型训练后的参数R
    # print(dt.classes_)
    # print(dt.feature_importances_)
    # print(dt.n_classes_)
    # print(dt.n_features_)
    # print(dt.tree_)
    # 绘制决策树（还要安装一个配置软件）
    # DTGraph(dt)


# 随机森林的oob分数
def r2(rf, X_train, y_train):
    return r2_score(y_train, rf.predict(X_train))


# 6_随机森林分类器
def RFModel(X_train, y_train, X_test, y_test, target):
    rf = RandomForestClassifier(random_state=42)
    rf.fit(X_train, y_train.values.ravel())
    score = rf.score(X_test, y_test)
    print('RF result：')
    print(score)
    print(rf.predict(target))
    print(rf.predict_proba(target))
    print(rf.predict_log_proba(target))
    # 特征重要性（gini划分）
    for col, val in sorted(
            zip(target.columns, rf.feature_importances_),
            key=lambda x: x[1],
            reverse=True,
    )[:5]:
        print(f'{col:10}{val:10.3f}')
    # 置换重要性（比特征重要性具有更好的度量）
    # perm_imp_rfpimp = permutation_importances(rf, X_train, y_train, r2)
    # print(perm_imp_rfpimp)


# 7_xgboost分类器
def XgbModel(X_train, y_train, X_test, y_test, target):
    xgb_class = xgb.XGBClassifier(random_state=42, early_stopping_rounds=10)
    xgb_class.fit(
        X_train,
        y_train.values.ravel(),
        eval_set=[(X_test, y_test.values.ravel())],
    )
    score = xgb_class.score(X_test, y_test.values.ravel())
    print('XGB result：')
    print(score)
    print(xgb_class.predict(target))
    print(xgb_class.predict_proba(target))
    # 输出特征重要性(根据节点的平均增益）
    for col, val in sorted(
            zip(
                target.columns,
                xgb_class.feature_importances_,
            ),
            key=lambda x: x[1],
            reverse=True,
    )[:5]:
        print(f"{col:10}{val:10.3f}")
    # 用xgb库绘制特征重要性图（其中importance_type参数标志不同的重要性度量,默认为weight）
    # fig,ax=plt.subplots(figsize=(6,4))
    # xgb.plot_importance(xgb_class,ax=ax)
    # fig.savefig('images/mlpr_1005.png',dpi=300)
    # xgb库绘制单独棵树(也是需要安装graphviz这个配置软件）
    # booster = xgb_class.get_booster()
    # print(booster.get_dump()[0])
    # fig,ax = plt.subplots(figsize=(6,4))
    # xgb.plot_tree(xgb_class,ax=ax,num_trees=0)
    # fig.savefig('images/mlpr_1007.png',dpi=300)
    # 使用xgbfir库实现多种特征重要性度量方法(通过该方法能很好地找到最适合的特征组合）
    # xgbfir.saveXgbFI(
    #     xgb_class,
    #     feature_names=target.columns,
    #     OutputXlsxFile='fir.xlsx',
    # )
    # pd.read_excel('fir.xlsx').head(3).T


# 8_lightgbm分类器
def lgbModel(X_train, y_train, X_test, y_test, target):
    lgbm_class = lgb.LGBMClassifier(random_state=42)
    lgbm_class.fit(X_train, y_train)
    score = lgbm_class.score(X_test, y_test)
    print('LGBM result：')

    print(score)
    print(lgbm_class.predict(target))
    print(lgbm_class.predict_proba(target))
    # 输出特征重要性（这里的特征重要性默认为'splits',按一个特征的使用次数,可以设置importance_type=gain换掉）
    for col, val in sorted(
            zip(target.columns, lgbm_class.feature_importances_),
            key=lambda x: x[1],
            reverse=True
    )[:5]:
        print(f'{col:10}{val:10.3f}')
    # 用lgbm库绘制特征重要性表
    # fig,ax = plt.subplots(figsize=(6,4))
    # lgb.plot_importance(lgbm_class,ax=ax)
    # fig.tight_layout()
    # fig.savefig('images/mlpr_1008.png',dpi=300)
    # 用lgbm库绘制lgbm树(也是需要安装graphiviz这个软件）
    # fig, ax = plt.subplots(figsize=(6,4))
    # lgb.plot_tree(lgbm_class,tree_index=0,ax=ax)
    # fig.savefig('images/mlpr_1009.png',dpi=300)


# 9_catboost分类器
def catboostModel(train, y_train, test, y_test, target):
    params = {'depth': [4, 7, 10],
              'learning_rate': [0.03, 0.1, 0.15],
              'l2_leaf_reg': [1, 4, 9],
              'iterations': [300]}
    cbc = cb.CatBoostClassifier()
    cb_model = GridSearchCV(cbc, params, scoring="roc_auc", cv=3)
    cv = cb_model.fit(train, y_train)
    print('模型最佳参数为：')
    print(cv.best_params_)

    clf = cb.CatBoostClassifier(eval_metric="AUC", depth=10, iterations=500, l2_leaf_reg=9, learning_rate=0.15)
    clf.fit(train, y_train)
    auc_value = auc(clf, train, y_train,test,y_test)
    print(auc_value)

    print(clf.predict(target))
    print(clf.predict_proba(target))

    # 有专家知识挑选出重要特征时使用
    # cat_features_index = [0, 1, 2, 3, 4, 5, 6]
    # clf = cb.CatBoostClassifier(eval_metric="AUC", one_hot_max_size=31, \
    #                             depth=10, iterations=500, l2_leaf_reg=9, learning_rate=0.15)
    # clf.fit(train, y_train, cat_features=cat_features_index)
    # auc(clf, train, test)

# 10_TPOT模型
def TPOTModel(X_train, y_train, X_test, y_test, target):
    tc = TPOTClassifier(generations=2)
    tc.fit(X_train, y_train)
    score = tc.score(X_test, y_test)
    print(score)
    print(tc.predict(target))
    print(tc.predict_proba(target))
    # 导出程序流水线
    # tc.export('tpot_exported_pipeline.py')
