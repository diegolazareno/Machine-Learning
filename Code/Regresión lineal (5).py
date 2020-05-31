# -*- coding: utf-8 -*-
"""
Created on Sat May 16 11:14:10 2020

@author: López Lazareno Diego Alberto 
"""

#%% Regresión lineal
# Parte 5. Regresión lineal con gradiente descendente

# Se importan las librerías necesarias
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use("ggplot")

# Lectura de archivos
data=pd.read_csv("../Data/ex1data1.txt",names=["population","profit"])
data["ones"]=1

# Inicializar los parámetros
X=np.array(data[["ones","population"]]) # Variable independiente
D=len(X[0,:]) # Número de componentes (pesos)
W=np.random.randn(D)/np.sqrt(D) # Pesos
Y=np.array(data["profit"]) # Variable dependiente
alpha=0.0001 # Learning rate
epochs=10000 # Iteraciones
n=len(Y) # Samples
cost=[] # MSE (error cuadrático medio)

# Descenso del gradiente
for i in range(epochs):
    yhat=np.dot(X,W) # Predicción
    delta=yhat-Y # Loss
    W=W-alpha*np.dot(X.T,delta) # Actualización de los pesos
    mse=np.dot(delta,delta)/n  # Error cuadrático medio por iteración
    cost.append(mse) 

# Visualización del error cuadrático medio por iteración
plt.figure(figsize=(5,5))
plt.xlabel("Iteración")
plt.ylabel("MSE")
plt.plot(cost,label="MSE",c="k")
plt.legend(loc="best")