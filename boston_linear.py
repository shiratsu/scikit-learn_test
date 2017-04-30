# coding:utf-8

import pylab
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split

# import some data to play with
boston = datasets.load_boston()
print(boston.DESCR)

# plt.hist(boston.target,bins=50)
# # 横軸のラベル名を設定
# plt.xlabel('Price in $1,000s')
# # 縦軸のラベル名を設定
# plt.ylabel('Number of houses')

# # ラベルがRMとなっている5番目の列が部屋数
# plt.scatter(boston.data[:,5],boston.target)

# plt.ylabel('Price in $1,000s')
# plt.xlabel('Number of rooms')

# DataFrameを作成
boston_df = pd.DataFrame(boston.data)
# 列名をつける
boston_df.columns = boston.feature_names
# 新しい列を作り目的変数である価格を格納
boston_df['Price'] = boston.target

# boston_df.head()

import sklearn
from sklearn.linear_model import LinearRegression
lreg = LinearRegression()

# 説明変数
X_multi = boston_df.drop('Price',1)

# 説明変数をX、目的変数をYとして格納
X_train, X_test, Y_train, Y_test = train_test_split(X_multi,boston_df.Price)

# 中身を確認
print(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)

# fit関数でモデル生成
lreg.fit(X_train,Y_train)
print(lreg.coef_)

# pred_train = lreg.predict(X_train)
# pred_test = lreg.predict(X_test)

# print('X_trainを使ったモデルの平均二乗誤差＝{:0.2f}'.format(np.mean((Y_train - pred_train) ** 2)))
# print('X_testを使ったモデルの平均二乗誤差＝{:0.2f}'.format(np.mean((Y_test - pred_test) ** 2)))

# # 学習用データの残差プロット
# train = plt.scatter(pred_train,(pred_train-Y_train),c='b',alpha=0.5)

# # テスト用データの残差プロット
# test = plt.scatter(pred_test,(pred_test-Y_test),c='r',alpha=0.5)

# # y=0の水平線
# plt.hlines(y=0,xmin=-10,xmax=50)

# plt.legend((train,test),('Training','Test'),loc='lower left')
