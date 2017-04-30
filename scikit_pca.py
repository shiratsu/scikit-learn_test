# coding:utf-8

import pylab
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets
from sklearn.decomposition import PCA
import numpy as np
# import some data to play with
iris = datasets.load_iris()
X = iris.data  # we only take the first two features.
Y = iris.target

# print(iris)
# print("----------------------------------------------------")
# print(X)
# print("----------------------------------------------------")
# print(Y)

# ----------------------------------------------------------
# pca = PCA(n_components = 3)
# print(pca.fit(X))
# # 主成分の次元ごとの寄与率を出力する
# print(pca.explained_variance_ratio_)

pca2 = PCA(n_components = 2)
print(pca2.fit(X))
print("------------fit----------------")
# 主成分の次元ごとの寄与率を出力する
print(pca2.explained_variance_ratio_)
print("-------------explained_variance_ratio_---------------")
total_kiyo = np.cumsum(pca2.explained_variance_ratio_)
print(total_kiyo)
print("--------------total_kiyo--------------")
print(pca2.components_)

# pca3 = PCA(n_components = 4)
# print(pca3.fit(X))
# # 主成分の次元ごとの寄与率を出力する
# print(pca3.explained_variance_ratio_)
x_pca=pca2.transform(X)

pylab.scatter(x_pca[:,0],x_pca[:,1])
