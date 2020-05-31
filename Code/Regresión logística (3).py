# -*- coding: utf-8 -*-
"""
Created on Tue May 19 11:37:28 2020

@author: López Lazareno Diego Alberto
"""

#%% Regresión logística 
# Parte 3. Cross entropy error function

# Se importan las librerías necesarias
import numpy as np 
import matplotlib.pyplot as plt

# Generar datos aleatorios (Gaussian Clouds)
N,D=100,2 
X=np.random.randn(N,D)
# Los primeros 50 datos centrados en (-2,-2)
X[:50,:]=X[:50,:]-2
# Los últimos 50 datos centrados en (2,2)
X[50:,:]=X[50:,:]+2

# Visualización
plt.figure(figsize=(8,5))
plt.scatter(X[:50,0],X[:50,1],c="salmon")
plt.scatter(X[50:,0],X[50:,1],c="darkcyan")
plt.scatter(np.array([[2],[-2]]),np.array([[2],[-2]]),c="k")
plt.grid()

# Target
T=np.ones((N))
T[:50]=0
T[50:]=1

# Variables independientes
ones=np.array([[1] for i in range(100)])
Xb=np.concatenate((ones,X),axis=1)

# Pesos
#w=np.random.randn(D+1)

# Función sigmoide
def sigmoid(X,W):
    Z=np.dot(X,W)
    return 1/(1+np.exp(-Z))

# Cross entropy error
def cross_entropy_error_f(T,Y):
    E=0
    for i in range(len(T)):
        if T[i]==0:
            E-=np.log(1-Y[i])
        else:
            E-=np.log(Y[i])
    return E

# Output de la sigmoide
#y=sigmoid(Xb,w)

# Error
#error=cross_entropy_error_f(T,y)

# Descenso del gradiente
alpha=0.1
epochs=1000
cost_iter=[]
w=np.random.randn(D+1)

for i in range(epochs):
    y=sigmoid(Xb,w)
    cost_iter.append(cross_entropy_error_f(T,y))
    gradient=np.dot(Xb.T,(y-T))
    w=w-alpha*gradient

#%% Con scikit-learn

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression()
lr.fit(Xb,T)
w_sk=lr.coef_ # Coeficientes
y_sklearn=lr.predict(Xb) # Predicción
error=cross_entropy_error_f(T,y_sklearn) # Error

# Con scklearn se obtiene un error de 0
# Con el algoritmo se obtiene un error cercano a 0

#%% Visualización del discriminante lineal encontrado 

x=np.linspace(-5,5,100)
y=-w[1]/w[2]*x-w[0]/w[2]
y_sk=-w_sk[0,1]/w_sk[0,2]*x-w_sk[0,0]/w_sk[0,2]
plt.figure(figsize=(8,5))
plt.grid()
plt.scatter(X[:,0],X[:,1],c=T,cmap="Pastel2",s=50)
plt.plot(x,y,c="salmon",label="Linear Discriminant (LR)")
plt.plot(x,y_sk,c="darkcyan",label="Linear Discriminant (Sklearn)")
plt.legend(loc="best")