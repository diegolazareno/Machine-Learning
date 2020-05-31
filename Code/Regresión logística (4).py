# -*- coding: utf-8 -*-
"""
Created on Tue May 19 22:40:04 2020

@author: López Lazareno Diego Alberto
"""

#%% Regresión logística 
# Parte 4. Proyecto e-commerce (final)

# Se importan las librerías necesarias
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Lectura de archivos
data=pd.read_csv("../Data/ecommerce_data.csv")

#%% Preprocesamiento 

# Variables independientes
X=data.iloc[:,:-1]
ones=np.array([[1] for i in range(len(X))])
# Variable dependiente
Y=data.iloc[:,-1]

# Normalizar las columnas con valores enteros
X.iloc[:,1]=(X.iloc[:,1]-np.mean(X.iloc[:,1]))/np.std(X.iloc[:,1])
X.iloc[:,2]=(X.iloc[:,2]-np.mean(X.iloc[:,2]))/np.std(X.iloc[:,2])

# Obtener variables dummy para la columna "time_of_day"
N,D=X.shape
dummy=np.zeros((N,4))

for i in range(len(X)):
    dummy[i,X.iloc[i,4]]=1
    
# Nuevas variables independientes
New_X=np.concatenate((ones,X.iloc[:,0:4],dummy),axis=1)
# Escoger las X's y Y's de únicamente 2 clases (0 o 1) en user_action
New_X=New_X[Y<=1]
New_Xframe=pd.DataFrame(New_X)
New_Y=Y[Y<=1]

# Train/Test
split=int(len(New_X)*0.7)
x_train=New_X[:split,:]
x_test=New_X[split:,:]
y_train=New_Y[:split]
y_test=New_Y[split:]

#%% Predicción

# Función sigmoide 
def sigmoid(X,W):
    Z=np.dot(X,W)
    return 1/(1+np.exp(-Z))

# Cross entropy error function
def cross_entropy(T,Y):
    return -np.mean(T*np.log(Y)+(1-T)*np.log(1-Y))

# Inicializar parámetros para el modelo
D=New_X.shape[1]
w=np.random.randn(D) # Pesos
alpha=0.001 # Learning Rate
epochs=10000 # Iteraciones
train_error=[] # Error en los datos de entrenamiento
test_error=[] # Error en los datos de prueba 

# Descenso del gradiente 
for i in range(epochs):
    yhat_train=sigmoid(x_train,w)
    yhat_test=sigmoid(x_test,w)
    train_error.append(cross_entropy(y_train,yhat_train))
    test_error.append(cross_entropy(y_test,yhat_test))
    w=w-alpha*np.dot(x_train.T,(yhat_train-y_train))

#%% Evaluación

# Exactitud del modelo
from sklearn.metrics import accuracy_score
y1=np.round(sigmoid(x_train,w))
y2=np.round(sigmoid(x_test,w))
print("Accuracy (Train)",accuracy_score(y_train,y1))
print("Accuracy (Test)",accuracy_score(y_test,y2))

# Visualización del coste por iteración
plt.figure(figsize=(10,5))
plt.plot(train_error,label="Train Error")
plt.plot(test_error,label="Test Error")
plt.grid()
plt.legend(loc="best")
plt.xlabel("Iteration")
plt.ylabel("Cost")