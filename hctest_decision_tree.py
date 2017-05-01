# coding:utf-8

import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
import skutil
from sklearn import tree
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
clf = tree.DecisionTreeClassifier()
clf.fit(X_train,y_train)
# print(len(X_train[0]))
# 重要度の予測
print(clf.feature_importances_)
# 予測
pred = clf.predict(X_test)

# 予測結果
print(pred)

# decision_path
print(clf.decision_path(X_test))

## レポート表示
print(metrics.classification_report(y_test,pred,target_names=['no','yes']))

# # print(clf.estimators_)
# i = 0
# for factor in clf.feature_importances_:
#     print('factor'+str(i)+':'+str(factor))
#     i += 1

from IPython.display import Image
import pydotplus
dot_data = tree.export_graphviz(clf, out_file=None,
                         filled=True, rounded=True,
                         special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data)
graph.write_pdf("retirement.pdf")

# ---------------------------------------------
# 結果を可視化してみる

n_nodes = clf.tree_.node_count
children_left = clf.tree_.children_left
children_right = clf.tree_.children_right
feature = clf.tree_.feature
threshold = clf.tree_.threshold

node_depth = np.zeros(shape=n_nodes)
is_leaves = np.zeros(shape=n_nodes, dtype=bool)
stack = [(0, -1)]  # seed is the root node id and its parent depth
while len(stack) > 0:
    node_id, parent_depth = stack.pop()
    node_depth[node_id] = parent_depth + 1

    # If we have a test node
    if (children_left[node_id] != children_right[node_id]):
        stack.append((children_left[node_id], parent_depth + 1))
        stack.append((children_right[node_id], parent_depth + 1))
    else:
        is_leaves[node_id] = True

print("The binary tree structure has %s nodes and has "
      "the following tree structure:"
      % n_nodes)
for i in range(n_nodes):
    if is_leaves[i]:
        # print("%snode=%s leaf node." % (node_depth[i] * "\t", i))
        print("-------------if node start--------------------")
        print(node_depth[i])
        print("-------------if node end--------------------")
    else:
        # print("%snode=%s test node: go to node %s if X[:, %s] <= %ss else to "
        #       "node %s."
        #       % (node_depth[i] * "\t",
        #          i,
        #          children_left[i],
        #          feature[i],
        #          threshold[i],
        #          children_right[i],
        #          ))
        print("-------------else node start--------------------")
        print(node_depth[i])
        print(i)
        print(children_left[i])
        print(feature[i])
        print(threshold[i])
        print(children_right[i])
        print("-------------else node end--------------------")
print()
