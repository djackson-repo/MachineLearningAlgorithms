#CON NEEDS LOTS OF TRAINING DATA

import pandas as pd
df=pd.read_csv("/Users/djackson/Downloads/Advertising.csv")
df.head(3)
import numpy as np
print(len(df))
data=df.to_numpy()
x=data[:,1:4]
y=data[:,4]
np.linalg.norm(x[0]-x[1])
#let K-3, find 3 closest points to X[0]
print(data.shape)
def dist(i,j):
    return np.linalg.norm(x[i]-x[j])

result=[(dist(0,0),0),(dist(0,1),1), (dist(0,2),2)]
for i in range(3,200):
    result.append((dist(0,i),i))
    result.sort()
    result.pop(-1)
    
print(result)
pred = 0
for i in result:
    pred+=y[i[1]]

pred = pred/3  
print(pred, y[0])

#MEANS SQUARED FUNCTION
def error(pred, actual):
    result = 0
    for i in range(len(actual)):
        result+=np.square(pred[i]-actual[i])
    
    return result/len(actual)

test = data[0:40, :]
train = data[40:, :]

x_train = train[:, 1:4]
y_train = train[:, 4]

x_test = test[:, 1:4]
y_test = test[:, 4]

def dist2(i,j):
    return np.linalg.norm(i-j)
    

k = 3
preds = []
for x in x_test:
    neigh = []
    for i in range(k):
        neigh.append((dist2(x_test, x_train[i]), i))
        
    for i in range(k, len(y_train)):
        neigh.append((dist2(x_test, x_train[i]), i))
        neigh.sort()
        neigh.pop(-1)
        
    for i in neigh:
        pred += y_train [i[1]]
    pred = pred / k
    preds.append(pred)


print(error(preds, y_test))
