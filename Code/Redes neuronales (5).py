# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 20:06:38 2020

@author: López Lazareno Diego Alberto 
"""

#%% Redes Neuronales
# Parte 5. Feedforward en una Red Neuronal y Backpropagation

#%% Preprocesamiento de los datos
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
plt.figure(figsize=(10,5))
plt.scatter(X[:,0],X[:,1],c=Y)

# Generar una matriz de NxK para Y (one-hot encoding)
N=len(Y)
K=max(Y)+1 # Número de clases (3)
y_=np.zeros((N,K))
for i in range(len(Y)):
    y_[i,Y[i]]=1

#%% Feedforward y Backpropagation
# Función feedforward
def feedforward(X,W1,b1,W2,b2):
    Z=1/(1+np.exp(-(np.dot(X,W1)+b1))) # Hidden layer
    A=np.dot(Z,W2)+b2 # Output layer
    expA=np.exp(A) # Softmax
    probA=expA/expA.sum(axis=1,keepdims=True) # Probabilidades de que cada sample pertenezca a la K-ésima clase
    return Z,probA
# Función classification_rate
def classification_rate(Y,P):
    n_correct=0
    n_total=len(Y)
    for i in range(len(Y)):
        if Y[i]==P[i]:
            n_correct=n_correct+1
    return n_correct/n_total

# Inicializar los parámetros del modelo
D=len(X[0]) # Dimensión de cada sample
M=3 # Número de unidades (neuronas) en la capa oculta
W1=np.random.randn(D,M) # Pesos (input layer to hidden layer)
b1=np.random.randn(M) # Bias (input layer to hidden layer)
W2=np.random.randn(M,K) # Pesos (hidden layer to output layer)
b2=np.random.randn(K) # Bias (hidden layer to output layer)

# Backpropagation 
epochs=100000 # Iteraciones
alpha=0.0001 # Learning rate
cost_per=[] # Costo por iteración

for i in range(epochs):
    Z,probA=feedforward(X,W1,b1,W2,b2) # Feedforward
    # Gradient ascend
    W2=W2+alpha*np.dot(Z.T,(y_-probA))
    b2=b2+alpha*np.sum((y_-probA),axis=0)
    Z_=Z*(1-Z)
    W1=W1+alpha*np.dot(X.T,(np.dot((y_-probA),W2.T)*Z_))
    b1=b1+alpha*np.sum(np.dot((y_-probA),W2.T)*Z_,axis=0)
    # Costo por iteración
    cost_per.append(np.sum(y_*np.log(probA)))
    
P=np.argmax(probA,axis=1) # Se escoge la posición cuya probabilidad es mayor (esto concuerda con las clases: 0, 1 y 2)
class_rate=classification_rate(Y,P) # Ratio de clasificación 

#%% Resultados
print("Classification rate:",class_rate)
plt.figure(figsize=(10,5))
plt.plot(cost_per)