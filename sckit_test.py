# coding:utf-8

import pandas as pd
from sklearn import tree

# サンプルデータ(タブ区切り)の読み込み
data = pd.read_table('Titanic_2.txt')
# 説明変数
variables = ['Class','Sex','Age']

# 決定木の分類器を作成。各種設定は引数に与える
classifier = tree.DecisionTreeClassifier()
# 決定木の分類器にサンプルデータを食わせて学習。目的変数はSpecies
classifier = classifier.fit(data[variables], data['Survived'])

# 学習した結果をGraphvizが認識できる形式にして出力する
with open('graph.dot', 'w') as f:
    f = tree.export_graphviz(classifier, out_file=f)
