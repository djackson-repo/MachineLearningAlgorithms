#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 10:14:31 2024

@author: djackson
"""
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
df=pd.read_csv("https://raw.githubusercontent.com/intro-stat-learning/ISLP/main/ISLP/data/Weekly.csv")

basicModel=DecisionTreeClassifier()
basicModel.fit(df[['Lag1', 'Lag2', 'Lag3', 'Lag4', 'Lag5', 'Volume']], df["Direction"])
(basicModel.predict(df[['Lag1', 'Lag2', 'Lag3', 'Lag4', 'Lag5', 'Volume']])==df["Direction"]).mean()

X=df[['Lag1', 'Lag2', 'Lag3', 'Lag4', 'Lag5', 'Volume']]
Y=df['Direction']

X_train=df.loc[df['Year']<2010, ['Lag1', 'Lag2', 'Lag3', 'Lag4', 'Lag5', 'Volume']]
X_test=df.loc[df['Year']==2010, ['Lag1', 'Lag2', 'Lag3', 'Lag4', 'Lag5', 'Volume']]

Y_train=Y[df['Year']<2010]
Y_test=Y[df['Year']==2010]

tree=DecisionTreeClassifier()
tree.fit(X_train, Y_train)
print((tree.predict(X_test)==Y_test).mean())

#Bagging Our Tree

bagging_tree=RandomForestClassifier(max_features=1.0, random_state=1)
bagging_tree.fit(X_train,Y_train)
print((bagging_tree.predict(X_test)==Y_test).mean())

#Random Forest

random_forest=RandomForestClassifier(random_state=1)
random_forest.fit(X_train, Y_train)
print((random_forest.predict(X_test)==Y_test).mean())

#Boosting

boosting_tree=GradientBoostingClassifier(learning_rate=0.1)
boosting_tree.fit(X_train, Y_train)
print((boosting_tree.predict(X_test)==Y_test).mean())