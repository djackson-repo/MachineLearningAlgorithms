

import pandas as pd
import numpy as np
from HW4 import LDA
from HW4 import QDA
df=pd.read_csv("https://raw.githubusercontent.com/intro-stat-learning/ISLP/main/ISLP/data/Weekly.csv")

X=df[['Lag2', 'Volume']]
Y=df['Direction']
k=6

predictionsQDA= (QDA(X, Y))
predictionsLDA= (LDA(X,Y))

print(pd.crosstab(predictionsQDA,Y))
print(np.mean(predictionsQDA==Y)) #Tells the percent of correctness
print(pd.crosstab(predictionsLDA,Y))
print(np.mean(predictionsLDA==Y)) #Tells the percent of correctness


folds=np.random.choice(range(X.shape[0]), size=(k, 1000), replace=True)

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

    predictions=LDA(x_train, y_train, x_test=x_test)
    error+=1-np.mean(predictions==y_test)

print("LDA error estimate is: ", round(error/k, ndigits=3))
predictions=LDA(X, Y)
error=1-(predictions==Y).mean()
print("Simpleton's LDA error estimate: ", round(error, ndigits=3))

folds=np.random.choice(range(X.shape[0]), size=(k, 1000), replace=True)

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

    predictions=QDA(x_train, y_train, x_test=x_test)
    error+=1-np.mean(predictions==y_test)

print("QDA error estimate is: ", round(error/k, ndigits=3))
predictions=QDA(X, Y)
error=1-(predictions==Y).mean()
print("Simpleton's QDA error estimate: ", round(error, ndigits=3))
