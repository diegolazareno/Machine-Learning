# -*- coding: utf-8 -*-
"""
Created on Sun Jun  7 11:18:37 2020

@author: López Lazareno Diego Alberto 
"""

#%% Redes Neuronales
# Parte 6. Proyecto E-commerce (clasificación multiclase con regresión logística)

# Se importan las librerías necesarias
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

# Semilla
np.random.seed(2001)

# Lectura de archivos
data=pd.read_csv("../Data/ecommerce_data.csv")

# Variables
X=data.iloc[:,:-1]
Y=data.iloc[:,-1]

# Normalizar los datos para las columnas: n_products_viewed y visit_duration
X.iloc[:,1]=(X.iloc[:,1]-X.iloc[:,1].mean())/X.iloc[:,1].std()
X.iloc[:,2]=(X.iloc[:,2]-X.iloc[:,2].mean())/X.iloc[:,2].std()

# Variables dummy para la columna time_of_day
X_dummy=np.zeros((len(X),X.iloc[:,-1].max()+1))
for i in range(len(X)):
    X_dummy[i,X.iloc[i,-1]]=1
    
New_X=np.concatenate((np.array(X.iloc[:,:4]),X_dummy),axis=1)

# One-hot encoding para la variable dependiente
K=np.max(Y)+1 # 4 clases
New_Y=np.zeros((len(Y),K))
for i in range(len(Y)):
    New_Y[i,Y.iloc[i]]=1

# Train/Test
split=int(len(New_X)*0.7)
x_train,x_test=New_X[:split],New_X[split:]
y_train,y_test=New_Y[:split],New_Y[split:]

# Funciones 
def forward(X,W,b):
    Z=np.dot(X,W)+b
    exp_Z=np.exp(Z)
    prob=exp_Z/exp_Z.sum(axis=1,keepdims=True)
    return prob
def cost(T,Y):
    return np.sum(T*np.log(Y))
def classification_rate(T,Y):
    n_total=len(T)
    n_correct=0
    for i in range(len(T)):
        if T.iloc[i]==Y[i]:
            n_correct=n_correct+1
    return n_correct/n_total

# Inicializar los parámetros del modelo
D=New_X.shape[1] # Dimensión de los pesos (8)
W=np.random.randn(D,K) # Pesos del modelo
b=np.random.randn(K) # Términos bias
alpha=0.0001 # Tasa de aprendizaje
epochs=100000 # Iteraciones
train_cost=[] # Costos por iteración en los datos de entrenamiento
test_cost=[] # Costos por iteración en los datos de prueba

# Ascenso del gradiente
for i in range(epochs):
    py_train=forward(x_train,W,b)
    py_test=forward(x_test,W,b)
    W=W+alpha*(np.dot(x_train.T,(y_train-py_train))) # Actualización de los pesos del modelo
    b=b+alpha*(np.sum(y_train-py_train,axis=0)) # Actualización de los bias del modelo
    # Costos
    train_cost.append(cost(y_train,py_train))
    test_cost.append(cost(y_test,py_test))
        
# Resultados
train_accu=classification_rate(Y.iloc[:split],np.argmax(py_train,axis=1)) # Accuracy para los datos de entrenamiento
test_accu=classification_rate(Y.iloc[split:],np.argmax(py_test,axis=1)) # Accuracy para los datos de prueba 
# Visualización
plt.figure(figsize=(10,5))
plt.plot(train_cost,label="Train cost") 
plt.plot(test_cost,label="Test cost")
plt.legend(loc="best")