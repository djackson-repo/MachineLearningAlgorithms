#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 10:48:11 2024

@author: djackson
"""
import pandas as pd
import numpy as np
#if using numpy this will work for x and y

df=pd.read_csv("/Users/djackson/Downloads/Advertising.csv")

def regression(x,y):
    top=0
    bottom=0
    x_mean=np.mean(x)
    y_mean=np.mean(y)
    for i in range(len(x)):
        top+=(x[i]-x_mean)*(y[i]-y_mean)
        bottom+=x[i]*(x[i]-x_mean)
    
    theta_1=top/bottom
    theta_0=y_mean-theta_1*x_mean
    return theta_0,theta_1

x=df['TV']
y=df['sales']

theta_0, theta_1 = regression(x,y)

import seaborn as sns
import matplotlib.pyplot as plt
sns.scatterplot(data=df,x='TV', y = 'sales')
plt.plot([0,300], [theta_0, 300*theta_1+theta_0], color="red")



var=theta_0
for Y in y:
    var+=np.square(Y-np.mean(y))
    
R_squared = (var-RSS)/var
print(R_squared)

sns.scatterplot(predictions-y)

plt.clf()
sns.scatterplot(x=x, y=predictions-y)

#SKLEARN STUFF

import sklearn.linear_model as linear_model
df['square_root_TV']= np.sqrt(df['TV'])
df['newspaper-radio']= df['newspaper']*df['radio']
model= linear_model.LinearRegression()
model.fit(df[['TV', 'radio', 'square_root_TV', 'newspaper-radio']], df['sales'])
print(model.coef_)
print(model.intercept_)
print(model.score(df[['TV', 'radio', 'square_root_TV','newspaper-radio']], df['sales']))

    