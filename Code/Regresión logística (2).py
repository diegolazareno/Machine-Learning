# -*- coding: utf-8 -*-
"""
Created on Mon May 18 15:37:41 2020

@author: López Lazareno Diego Alberto
"""

#%% Regresión logística 
# Parte 2. Proyecto e-commerce

# Se importan las librerías necesarias
import pandas as pd
import numpy as np

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

#%% Predicción

# Inicializar pesos aleatorios para el modelo
N,D=New_X.shape
w=np.random.randn(D)

# Función sigmoide (neurona)
def sigmoid(X,W):
    Z=np.dot(X,W)
    return 1/(1+np.exp(-Z))

# Output de la neurona
output=sigmoid(New_X,w)

# Ratio de clasificación
output_round=np.round(output) # Se redondea el output (1's y 0's)
classification=np.mean(New_Y==output_round)