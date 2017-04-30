# coding:utf-8

import pylab
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets
from sklearn.decomposition import PCA
import numpy as np
import pylab
import pandas as pd

data = pd.read_csv( 'HC.test.161228.v2.eng.csv', dtype={'hiring.source':'category','sex':'category'})
print(data)
data['hiring.source'] = data['hiring.source'].cat.codes
print(data)
data['sex'] = data['sex'].cat.codes
print(data)
