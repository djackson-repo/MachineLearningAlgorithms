#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 10:06:05 2024

@author: djackson
"""

import pandas as pd
import numpy as np
from LDA_func import LDA_binary

df=pd.read_csv("https://raw.githubusercontent.com/intro-stat-learning/ISLP/main/ISLP/data/Default.csv")

X=df[['income', 'balance']]
Y=df['default']

k=10

folds=np.random.choice(range(X.shape[0]), size=(k, 1000), replace=False)

error=0
for i in range(k):
    x_test= X.iloc[folds[i]]
    y_test= Y.iloc[folds[i]]
    train_index=[]
    for j in range(k):
        if i != j:
            train_index=np.concatenate((train_index, folds[j]))
    y_train=Y.iloc[train_index]
    x_train=X.iloc[train_index]

    predictions=LDA_binary(x_train, y_train, x_test=x_test)
    error+=1-np.mean(predictions==y_test)

print("error estimate is: ", round(error/k, ndigits=3))
predictions=LDA_binary(X, Y)
error=1-(predictions==Y).mean()
print("Simpleton's error estimate: ", round(error, ndigits=3))
