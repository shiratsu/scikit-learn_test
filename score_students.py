# coding:utf-8

import pylab
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets
from sklearn.decomposition import PCA
import numpy as np
import pylab

data = np.loadtxt('score.csv',delimiter=',')
name = ['tanaka','sato','suzuki','honda','kawabata','yoshino','saito']

pca2 = PCA(n_components = 2)
pca2.fit(data)
print(pca2)
print(pca2.components_)
print("-----------------------------------------------")
print(np.cumsum(pca2.explained_variance_ratio_))
print("-----------------------------------------------")
x = pca2.transform(data)
print(x)
print("-----------------------------------------------")
pyplot.ion()
pyplot.clf()

colors = [pyplot.cm.hsv(0.1*i,1) for i in range(len(name))]
for i in range(len(name)):
    pyplot.scatter(x[i,0],x[i,1],c=colors[i],label=name[i])
    pyplot.legend()

# pylab.scatter(x[:,0],x[:,1])
