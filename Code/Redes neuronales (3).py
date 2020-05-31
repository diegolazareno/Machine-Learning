# -*- coding: utf-8 -*-
"""
Created on Thu May 28 13:12:59 2020

@author: López Lazareno Diego Alberto
"""

#%% Redes Neuronales
# Parte 3. Feedforward en una Red Neuronal para clasificación binaria 

# Se importan las librerías necesarias 
import numpy as np
import matplotlib.pyplot as plt 

# Nubes gaussianas 
N=200 # Samples
D=2 # Dimensión de los samples
X=np.random.randn(200,2)
# Nube centrada en (2,2)
X[:100,:]=X[:100,:]+2
# Nube centrada en (-2,-2)
X[100:,:]=X[100:,:]-2
# Clases
Y=np.array([0]*100+[1]*100)

# Visualización
plt.scatter(X[:,0],X[:,1],c=Y)

#%% Clasificación binaria con la función softmax

# Funciones 
def forward(X,W1,b1,W2,b2):
    Z=1/(1+np.exp(-(np.dot(X,W1)+b1))) # Sigmoide
    A=np.dot(Z,W2)+b2
    exp_A=np.exp(A) # Función softmax
    prob_A=exp_A/exp_A.sum(axis=1,keepdims=True)
    return prob_A

def classification_rate(Y,T):
    n_total=len(T)
    n_correct=0
    for i in range(len(T)):
        if T[i]==Y[i]:
            n_correct=n_correct+1
    return n_correct/n_total

# Inicializar parámetros del modelo
D=2  # Dimensión de cada sample
M=3 # 3 nodos ocultos 
K=2 # 2 clases
W1=np.random.randn(D,M) 
b1=np.random.randn(M)
W2=np.random.randn(M,K)
b2=np.random.randn(K)

py=forward(X,W1,b1,W2,b2)
y=np.argmax(py,axis=1)
class_rate=classification_rate(y,Y) # Class rate esperado de 1/2

#%% Clasificación binaria con la función sigmoide (última capa)

# Función
def forward_sigmoid(X,W1,b1,W2,b2):
    Z=np.dot(X,W1)+b1
    tanh_Z=np.tanh(Z)  # Tangente hiperbólica
    A=np.dot(tanh_Z,W2)+b2
    prob_A=1/(1+np.exp(-A)) # Sigmoide
    return prob_A

# Inicializar parámetros del modelo
D_=2 # Dimensión de cada sample
M_=3 # 3 nodos ocultos
K_=1 # 1 clase 
W1_=np.random.randn(D_,M_)
b1_=np.random.randn(M_)
W2_=np.random.randn(M_)
b2_=np.random.randn(K_)

py_=forward_sigmoid(X,W1_,b1_,W2_,b2_)
y_=np.round(py_)
class_rate_=classification_rate(y_,Y) # Class rate esperado de 1/2