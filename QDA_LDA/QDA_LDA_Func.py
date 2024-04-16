#CON
#LINEAR BOUNDARY OBSERVATIONS NORMALLY DISTRIBUTED


import pandas as pd
import numpy as np
df=pd.read_csv("https://raw.githubusercontent.com/intro-stat-learning/ISLP/main/ISLP/data/Weekly.csv")

def LDA(X,Y, *args, **kwargs):
    
    X_test=kwargs.get('x_test', X)

    values=pd.unique(Y)
    rho0=len(Y[Y==values[0]])/len(Y)
    rho1=1-rho0
    
    m0=X[Y==values[0]].mean()
    m1=X[Y==values[1]].mean()
    
    A=np.concatenate((X[Y==values[0]]-m0, X[Y==values[1]]-m1))
    
    var= 1/(len(Y)-2)*np.matmul(A.T, A)
    
    const0=np.log(rho0) + -1/2*np.matmul(m0.T, np.matmul(np.linalg.inv(var),m0))
    coef0=np.matmul(np.linalg.inv(var), m0)
    
    const1=np.log(rho1) + -1/2*np.matmul(m1.T, np.matmul(np.linalg.inv(var),m1))
    coef1=np.matmul(np.linalg.inv(var), m1)
    
    
    
    predictions=[]
    
    for i in range(X_test.shape[0]):
        x=X_test.iloc[i]
        d0=np.matmul(x, coef0)+const0
        
        d1=np.matmul(x, coef1)+const1
        
        if d0>d1:
            predictions.append(values[0])
        else:
            predictions.append(values[1])
            
    return predictions

def QDA(X,Y, *args, **kwargs):
    
    X_test=kwargs.get('x_test', X)

    values=pd.unique(Y) ## GETS UNIQUE VALUES AND MAPS THEM TO NUMBER
    rho0=len(Y[Y==values[0]])/len(Y)
    rho1=1-rho0
    
    m0=X[Y==values[0]].mean()
    m1=X[Y==values[1]].mean()
    
    A0=X[Y==values[0]]-m0
    A1=X[Y==values[1]]-m1
    
    var0 = np.matmul(A0.T, A0) / (len(Y[Y == values[0]]) - 1)
    var1 = np.matmul(A1.T, A1) / (len(Y[Y == values[1]]) - 1)

    const0=np.log(rho0) + (-1/2)*np.matmul(m0.T, np.matmul(np.linalg.inv(var0),m0))
    coef0=np.matmul(np.linalg.inv(var0), m0)
    
    const1=np.log(rho1) + (-1/2)*np.matmul(m1.T, np.matmul(np.linalg.inv(var1),m1))
    coef1=np.matmul(np.linalg.inv(var1), m1)
    
    
    
    predictions=[]
    
    for i in range(X_test.shape[0]):
        x = X_test.iloc[i]

        d0=(-1/2)*(np.matmul(x.T, np.matmul(np.linalg.inv(var0), x)) + np.matmul(x, coef0) + const0)
        d1=(-1/2)*(np.matmul(x.T, np.matmul(np.linalg.inv(var1), x)) + np.matmul(x, coef1) + const1)
        
        if d0>d1:
            predictions.append(values[0])
        else:
            predictions.append(values[1])
            
    return predictions