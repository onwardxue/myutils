# myutils
这是我个人学习使用的软件包，包含各种有助于本人毕业论文的代码

目前有三个目录：
1.data_set(数据集)
  (1)credit_card_clients(信用卡借贷)

2.data_process(数据处理)
  (1)execute.py(主要执行文件，主)
  (2)preprocess_utils.py(数据预处理工具)
    -1- getData(读取数据)
    -2- dataInf(查看数据集整体信息)
    -3- del_feature(删除特征)
    -4- type_change(类别变量转数值类型)
    -5- extract_label(提出标签属性)
    -6- train_test(数据集划分为训练集和测试集)
    -7- deal_none(缺失值填充)
    -8- dataRegular(数值特征归一化)
    -9- multiModel(使用8种分类器进行预测，交叉检测)
    -10- modelAccess(模型评估，重要特征排序)
    -11- saveModel(保存模型)
    -12- importModel(导入模型)
    -13- model_predict(模型预测)
    -14- modelOpt(模型优化-网格搜索，基于随机森林)
    -15- plotMatrix(绘制混淆矩阵)
    -16- plotLearning(绘制学习曲线)
    -17- histogram(查看某个特征的数值分布直方图)
    -18- RFECV_screen(使用rfecv进行特征重要性排序)
    -19- multiinfo_screen(使用互信息进行特征重要性排序)

  (3)model_utils.py(分类器工具-9个)
    -1- inv_logit(计算反对数)
    -2- auc(计算auc)
    -3- logisticRegressModel(逻辑回归分类器)
    -4- GaussianNBModel(朴素贝叶斯分类器)
    -5- SVMModel(SVM分类器)
    -6- knnModel(knn分类器)
    -7- DTModel(决策树分类器)
    -8- RFModel(随机森林分类器)
    -9- XgbModel(xgboost分类器)
    -10- lgbModel(lightgbm分类器)
    -11- catboostModel(catboost分类器)
    -12- TPOTModel(TPOT分类器)？有问题
    ...(还可加集成学习几种方法-voting/bagging/adaboost
        /gdbt(gradiantboostingclassifier/histgradient..))

  (4)PCA_utils.py(PCA降维工具)
    -1- pcaTest(PCA降维测试，主)
    -2- screePlot(根据成分方差解释率绘制碎石图)
    -3- variblePlot(绘制碎石图)
    -4- relatePlot(查看各特征对主成分的影响)
    -5- ...

  (5)postprocess_utils.py(后处理工具)
    -1- confusion_matrix_plot(混淆矩阵)
    -2- report(分类报告)
    -3- rocCurve(绘制roc曲线)
    -4- apCurve(绘制精确率-召回率曲线)
    -5- cbPlot(绘制柱形图查看各类数量)
    -6- ...

  (6)Explaining_utils.py(可解释性工具) - 待改
    -1- limePlot(lime模型局部解释-显示每个特征对结果的影响)？不能用
    -2- treeExplain(树模型的可解释-显示每个特征对分类的影响)
    -3- pdboxPlot(部分依赖图-查看特征值的变化对结果的影响)？有问题
    -4- replaceModel(替代模型-用可解释模型替不可解释模型进行结果解释)
    -5- shapTest(shap值解释结果-用特征shaply值的可加性解释模型的预测结果)
    ..(还有回归模型的截距、回归系数、skl树模型的.feature_importances)

3.config(配置)
  (1)config_1.py(实现将print输出到控制台的内容一并写入文件中)

