# coding:utf-8

import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
import skutil
from sklearn import ensemble
from sklearn import metrics

# data = pd.read_csv( 'HC.test.161228.v2.eng.csv', dtype={'hiring.source':'category','sex':'category','retirement':'category'})
# データの読み込み
data = pd.read_csv("HC.test.161228.v2.eng.csv", dtype={'team':'category','biz.unit':'category','hiring.source':'category','sex':'category','retirement':'category','location':'category','education':'category','job.group':'category'})
# print(data.dtypes)
# print(bank.head(3))
# data['hiring.source'] = data['hiring.source'].cat.categories
# i = data['hiring.source'].cat.categories

# data['sex'] = data['sex'].cat.categories
# data['location'] = data['location'].cat.categories
# data['education'] = data['education'].cat.categories
# data['job.group'] = data['job.group'].cat.categories
cat_columns = data.select_dtypes(['category']).columns
data[cat_columns] = data[cat_columns].apply(lambda x: x.cat.codes)

data = data.drop('name',1)
data = data.drop('date.of.birth',1)
data = data.drop('joining.date',1)
data = data.drop('termination.date',1)
print(data)
features,label = data.drop('retirement',1),data.retirement
# print(features)
x_features = features.as_matrix()
y_features = label.as_matrix()
print(x_features[0])
# print(x_features)
# print(y_features[0])

random_state = np.random.RandomState(123)
X_train,X_test,y_train,y_test = train_test_split(x_features,y_features,test_size=.3,random_state=random_state)

# RandomForestによるyosoku
# clf = ensemble.RandomForestClassifier(n_estimators=500,random_state=random_state)
clf = ensemble.RandomForestClassifier()
clf.fit(X_train,y_train)
# print(len(X_train[0]))
print(len(clf.feature_importances_))
print(clf.feature_importances_)
# 予測
pred = clf.predict(X_test)

print(metrics.classification_report(y_test,pred,target_names=['no','yes']))

# # print(clf.estimators_)
# i = 0
# for factor in clf.feature_importances_:
#     print('factor'+str(i)+':'+str(factor))
#     i += 1
