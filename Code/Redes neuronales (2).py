# -*- coding: utf-8 -*-
"""
Created on Thu May 28 12:11:45 2020

@author: López Lazareno Diego Alberto 
"""

#%% Redes Neuronales
# Parte 2. Feedforward en una Red Neuronal

# Se importan las librerías necesarias
import numpy as np
import matplotlib.pyplot as plt

# Gaussian clouds
n=500 # Samples
# Nube centrada en (0,-2)
x1=np.random.randn(n,2)
x1[:,0]=x1[:,0]+0
x1[:,1]=x1[:,1]-2
# Nube centrada en (2,2)
x2=np.random.randn(n,2)
x2[:,0]=x2[:,0]+2
x2[:,1]=x2[:,1]+2
# Nube centrada en (-2,2)
x3=np.random.randn(n,2)
x3[:,0]=x3[:,0]-2
x3[:,1]=x3[:,1]+2

# Variables independientes
X=np.concatenate((x1,x2,x3),axis=0)
# Clases (3)
Y=np.array([0]*n+[1]*n+[2]*n)
# Visualización
plt.scatter(X[:,0],X[:,1],c=Y)

# Inicializar los parámetros del modelo
D=2 # Dimensión de los samples
M=3 # Hidden layer (3 unidades)
K=3 # Clases (3)
# Hidden layer
W1=np.random.randn(D,M) # Pesos
b1=np.random.randn(M) # Bias
# Output layer
W2=np.random.randn(M,K) # Pesos
b2=np.random.randn(K) # Bias

# Función forward
def forward(X,W1,b1,W2,b2):
    Z=1/(1+np.exp(-(np.dot(X,W1)+b1))) # Capa oculta (utiliza la sigmoide como no-linealidad)
    A=np.dot(Z,W2)+b2  # Capa de salida  
    expA=np.exp(A) # Función softmax
    Y=expA/expA.sum(axis=1,keepdims=True) # Probabilidades de que cada sample pertenezca a la K-ésima clase
    return Y 

# Función classification_rate
def classification_rate(Y,P):
    n_correct=0
    n_total=len(Y)
    for i in range(len(Y)):
        if Y[i]==P[i]:
            n_correct=n_correct+1
    return n_correct/n_total

P_Y_given_X=forward(X,W1,b1,W2,b2)
P=np.argmax(P_Y_given_X,axis=1) # Se escoge la posición cuya probabilidad es mayor (esto concuerda con las clases: 0, 1 y 2)
class_rate=classification_rate(Y,P) # Classification rate esperada de 1/3