# -*- coding:utf-8 -*-
# @Time : 2022/7/24 4:38 下午
# @Author : Bin Bin Xue
# @File : other_utils
# @Project : myutils
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 1_PCA降维
def pcaTest(X):
    pca = PCA(random_state=42)
    X_pca = pca.fit_transform(
        StandardScaler().fit_transform(X)
    )
    # 输出每个成分的方差解释率
    ratio = pca.explained_variance_ratio_
    print(ratio)
    # 输出主成分
    print(pca.components_[0])
    # 将方差解释率累计情况绘制成碎石图（显示主成分所含信息量，用肘部方法查看它的拐点，从而决定使用多少个主成分）
    # 由图中可知，仅保留3个特征就能保留90%的有效信息（方差）
    # screePlot(ratio)
    # 将方差解释率绘制成方差累计图
    # variblePlot(ratio)
    # # 绘制特征与主成分的关系图，查看每个特征对主成分的影响
    # ralatePlot(X,pca)
    # 绘制特征与主成分之间关系的柱状图
    # barPlot(pca)
    # 如果有很多特征，我们想要限制特征数，用代码找出前两个主成分中，权重绝对值至少为0.5的特征
    # lmfeaPlot(pca)


    # 根据成分的方差解释率绘制碎石图
    def screePlot(ratio):
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(ratio)
        ax.set(
            xlabel='Component',
            ylabel='Percent of Explained variance',
            title='Scree plot',
            ylim=(0, 1),
        )
        fig.savefig(
            'images/mlpr_1701.png',
            dpi=300,
            bbox_inches='tight',
        )

# 绘制方差累计图
def variblePlot(ratio):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(
        np.cumsum(ratio)
    )
    ax.set(
        xlabel='Component',
        ylabel='Percent of Explained variance',
        title='cumlative variance',
        ylim=(0, 1)
    )
    fig.savefig('images/mlpr_1702.png', dpi=300)

# 查看各特征对主成分的影响
def ralatePlot(X,pca):
    fig, ax = plt.subplots(figsize=(6, 4))
    plt.imshow(
        pca.components_.T,
        cmap='Spectral',
        vmin=-1,
        vmax=1,
    )
    plt.yticks(range(len(X.columns)), X.columns)
    plt.xticks(range(8), range(1, 9))
    plt.xlabel('principal component')
    plt.ylabel('Contribution')
    plt.title(
        'Contribution of Features to Components'
    )
    plt.colorbar()
    plt.savefig('images/mlpr_1703.png', dpi=300)

# 绘制柱状图
def barPlot(X,pca):
    fig, ax = plt.subplots(figsize=(8, 4))
    pd.DataFrame(
        pca.components_, columns=X.columns
    ).plot(kind='bar', ax=ax).legend(
        bbox_to_anchor=(1, 1)
    )
    fig.savefig('images/mlpr_1704.png', dpi=300)


# 筛选出较为重要的特征
def lmfeaPlot(X,pca):
    comps = pd.DataFrame(
        pca.components_, columns=X.columns
    )
    min_val = 0.5
    num_components = 2
    pca_cols = set()
    for i in range(num_components):
        parts = comps.iloc[i][
            comps.iloc[i].abs() > min_val
            ]
        pca_cols.update(set(parts.index))
    print(pca_cols)