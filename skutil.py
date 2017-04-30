# coding: UTF-8

import numpy as np
import pandas as pd

def makeFeatures(x):

    # 数値変数のスケーリング
    cn_num = ['age'
            ,'tenure'
            ,'execution.score'
            ,'cognitive.score'
            ,'social.intelligence'
            ,'compensation'
            ,'OT.hour'
            ,'joining.year'
            ,'performance.score'
            ,'GPA'
            ,'manager.cognitive.score'
            ,'manager.social.intelligence'
            ,'manager.execution.score'
            ,'engagement']
    x_num = x[cn_num]

    x[cn_num] = (x_num-x_num.mean())/x_num.std()

    # ダミー変数の変換
    x_dum = pd.get_dummies(x)
    return x_dum
