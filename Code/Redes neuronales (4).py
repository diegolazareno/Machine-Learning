# -*- coding: utf-8 -*-
"""
Created on Thu May 28 14:53:05 2020

@author: López Lazareno Diego Alberto 
"""

#%% Redes Neuronales
# Parte 4. Proyecto E-commerce (preprocesamiento de los datos y predicciones sin entrenamiento del modelo)

# Se importan las librerías necesarias
import numpy as np 
import pandas as pd

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

# Sólo tomar los samples que pertenecen a las clases 0 y 1
New_X=New_X[Y<=1]
New_Y=np.array(Y[Y<=1])

# Funciones
def forward(X,W1,b1,W2,b2):
    Z=np.tanh(np.dot(X,W1)+b1)
    A=np.dot(Z,W2)+b2
    EXP_A=np.exp(A)
    PROB_A=EXP_A/EXP_A.sum(axis=1,keepdims=True)
    return PROB_A

def classification_rate(T,Y):
    n_total=len(T)
    n_correct=0
    for i in range(len(T)):
        if T[i]==Y[i]:
            n_correct=n_correct+1
    return n_correct/n_total

# Inicializar los parámetros del modelo
D=len(New_X[0]) # Dimensión de cada sample
K=max(New_Y)+1 # Clases 
M=5 # 5 unidades en la capa escondida
W1=np.random.randn(D,M)
b1=np.random.randn(M)
W2=np.random.randn(M,K)
b2=np.random.randn(K)

# Resultados del modelo
py=forward(New_X,W1,b1,W2,b2)
y=np.argmax(py,axis=1)
class_rate=classification_rate(New_Y,y)