# -*- coding:utf-8 -*-
# @Time : 2022/7/24 9:13 下午
# @Author : Bin Bin Xue
# @File : Explaining_utils
# @Project : myutils

# 这里介绍几种模型可解释性的指标

from lime import lime_tabular
from treeinterpreter import(
    treeinterpreter as ti
)
# 问题？安装不了这个包
# import pdpbox

from sklearn.ensemble import RandomForestClassifier

from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
import shap


# lime 模型局部解释
# def limePlot(dt):
#     explainer = lime_tabular.LimeTabularExplainer(
#         X_train.values,
#         feature_names=X.columns,
#         class_names=['died','survived']
#     )
#     exp = explainer.explain_instance(
#         X_train.iloc[-1].values, dt.predict_proba
#     )
    # 绘图
    # fig = exp.as_pyplot_figure()
    # fig.tight_layout()
    # fig.savefig('images/mlpr_1301.png')
    # # 从图中可以看出性别特征对决策起到了重要作用（若不分性别，预测的死亡率为48%；若筛选为男性，则预测死亡率为80%）
    # data = X_train.iloc[-2].values.copy()
    # dt.predict_proba(
    #     [data]
    # )
    # data[5]=1
    # dt.predict_proba([data])

# 树模型的可解释
def treeExplain(dt,X):
    # 前两个
    instances = X.iloc[:4]
    prediction,bias,contribs = ti.predict(
        dt,instances
    )
    for i in range(4):
        print('Instance',i)
        print('Prediction',prediction[i])
        print('Bias (trainset mean)', bias[i])
        print('Feature contributions:')
        for c , feature in zip(
            contribs[i],instances.columns
        ):
            print('{}{}'.format(feature,c))

# 部分依赖图 ？包加载不了
# def pdboxPlot():
    # rf5 = RandomForestClassifier(
    #     **{
    #         'max_features':'auto',
    #         'min_samples_leaf':'0.1',
    #         'n_estimators':'200',
    #         'random_state':42,
    #     }
    # )
    # rf5.fit(X_train,y_train)
    # feat_name='Age'
    # p = pdp.pdp_iso
    # # 可视化两个特征的交互作用
    # features = ['Fare','Sex_male']
    # p = pdp_interact(
    #     rf5
    # )

# 替代模型（使用决策树模型解释svm模型）
def replaceModel(X_train,X_test,y_train,y_test):
    # 使用svm训练模型（这里可以换成其他不可解释模型！）
    sv = svm.SVC()
    sv.fit(X_train,y_train.values.ravel())
    # 使用测试集和svm预测结果数据训练决策树模型，通过决策树模型输出特征重要性
    sur_dt = DecisionTreeClassifier()
    sur_dt.fit(X_test,sv.predict(X_test))
    # 输出特征重要性
    for col, val in sorted(
        zip(
            X_test.columns,
            sur_dt.feature_importances_,
        ),
        key = lambda x: x[1],
        reverse=True,
    )[:7]:
        print(f'{col:10}{val:10.3f}')

# shapley值，可以解释任何机器学习模型的输出
def shapTest(model,X_train,y_train,X_test):
    # rf5 = RandomForestClassifier(
    #     **{
    #         'min_samples_leaf':0.1,
    #         'n_estimators':200,
    #         'random_state':42,
    #     }
    # )
    model.fit(X_train,y_train.values.ravel())
    # s = shap.TreeExplainer(rf5)
    s = shap.TreeExplainer(model)
    shap_vals = s.shap_values(X_test)
    target_idx = 1
    shap.force_plot(
        s.expected_value[target_idx],
        shap_vals[target_idx][20,:],
        feature_names=X_test.columns,
        matplotlib = True
    )
    # 验证预测均值
    y_base = s.expected_value
    print(y_base)

    # pred = model.predict(model.DMatrix(X_test)) # ？出问题，不通用，只有xgb有这个DMatrix方法
    # print(pred.mean())
    # 为每个样本绘制其每个特征的SHAP值，这可以更好地理解整体模式，并允许发现预测异常值。每一行代表一个特征，横坐标为SHAP值。
    # 一个点代表一个样本，颜色表示特征值(红色高，蓝色低)
    shap.summary_plot(shap_vals, X_test)
    # 之前提到传统的importance的计算方法效果不好，SHAP提供了另一种计算特征重要性的思路。
    # 取每个特征的SHAP值的绝对值的平均值作为该特征的重要性，得到一个标准的条形图
    shap.summary_plot(shap_vals, X_test, plot_type="bar")
    #interaction value是将SHAP值推广到更高阶交互的一种方法。树模型实现了快速、精确的两两交互计算，
    # 这将为每个预测返回一个矩阵，其中主要影响在对角线上，交互影响在对角线外。这些数值往往揭示了有趣的隐藏关系(交互作用)
    # shap_interaction_values = s.shap_interaction_values(X_test)
    # shap.summary_plot(shap_interaction_values, X_test)  #？出问题
    #为了理解单个feature如何影响模型的输出，我们可以将该feature的SHAP值与数据集中所有样本的feature值进行比较。
    # 由于SHAP值表示一个feature对模型输出中的变动量的贡献，下面的图表示随着特征RM变化的预测房价(output)的变化。
    # 单一RM(特征)值垂直方向上的色散表示与其他特征的相互作用，为了帮助揭示这些交互作用，“dependence_plot函数”自动选择另一个用于着色的feature。
    shap.dependence_plot("PAY_4", shap_vals[0], X_test)


if __name__ == '__main__':
    dt = DecisionTreeClassifier(random_state=42, max_depth=3)
    # dt.fit(X_train, y_train)
    # 1.回归系数
    # 截距和回归系数解释了模型的预测结果以及特征对结果的影响。回归系数为正表示正相关。
    # 2.特征重要性
    # skl的树模型都带有一个.feature_importances_属性，可用于显示特征重要性
    # 3.模型局部解释（LIME包) - 显示每个特征对决策正负的影响大小
    # limePlot(dt)
    # 4.解释树模型（包括决策树、随机森林、极限树）
    # 用treeinterpreter包解释，会给出每个特征给每个类的贡献列表(本例中显示年龄和性别的贡献最大）
    # treeExplain(dt)
    # 5.部分依赖图（查看特征值的变化是如何影响结果的）
    # 例子：用pdbox包绘制年龄对乘客死亡的影响
    # pdboxPlot()
    # 6.替代模型（用可解释的模型替代不可解释的模型）
    # svm和神经网络一般不可解释
    # 例子：用决策树模型替代支持向量机模型进行解释（先用训练集和标签训练svm模型，用测试集数据和svm预测测试集数据得到的标签训练决策树模型）
    # replaceModel()
    # 7.shapley值（SHAP包，能生成shap值，利用特征shaply值的可加性解释模型的预测结果，能可视化任意模型的特征贡献）
    # 例子：
    # shapTest()
    print('sss')