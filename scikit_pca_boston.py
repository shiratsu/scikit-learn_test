# coding:utf-8

import pylab
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd

# import some data to play with
boston = datasets.load_boston()
# print(boston.DESCR)
X = boston.data  # we only take the first two features.
Y = boston.target

# print(pd.DataFrame(boston.data, columns=boston.feature_names))



pca2 = PCA(n_components = 2)
pca2.fit(X)
# print(pca2.fit(X))
# # print("------------fit----------------")
# # 主成分の次元ごとの寄与率を出力する
# print(pca2.explained_variance_ratio_)
# print("-------------explained_variance_ratio_---------------")
# total_kiyo = np.cumsum(pca2.explained_variance_ratio_)
# print(total_kiyo)
# print("--------------total_kiyo--------------")
# print(pca2.components_)

# # pca3 = PCA(n_components = 4)
# # print(pca3.fit(X))
# # # 主成分の次元ごとの寄与率を出力する
# # print(pca3.explained_variance_ratio_)
x_pca=pca2.transform(X)
print(x_pca)
pylab.scatter(x_pca[:,0],x_pca[:,1])
