# -*- coding: utf-8 -*-
"""
Created on Sat May 16 11:09:51 2020

@author: López Lazareno Diego Alberto 
"""

#%% Regresión lineal
# Parte 4. Regresión lineal (con regularización de Ridge L2)

# Se importan las librerías necesarias
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use("ggplot")

# Lectura de archivos
data=pd.read_csv("../Data/ex1data1.txt",names=["population","profit"])
data["ones"]=1

# Variable independiente
x=np.array(data[["ones","population"]])
# Variable dependiente
y=np.array(data["profit"])

# Regresión lineal sin regularización
w_1=np.linalg.solve(np.dot(x.T,x),np.dot(x.T,y))
yhat=np.dot(x,w_1)

#%% Regresión de Ridge
# Se evitan parámetros grandes para el modelo añadiendo una penalización 
# Esto ayuda a reducir el sobreajuste por los outliers
lamb=1000 # Constante lambda para la penalización (hiperparámetro)
d=len(x[0,:]) # Dimensión de los pesos para la matriz identidad
w_2=np.linalg.solve(lamb*np.eye(d)+np.dot(x.T,x),np.dot(x.T,y))
yhat_r=np.dot(x,w_2)

# Visualización
plt.figure(figsize=(10,5))
plt.xlabel("population")
plt.ylabel("profit")
plt.scatter(x[:,1],y,c="k")
plt.plot(x[:,1],yhat,label="Regresión lineal",c="gold")
plt.plot(x[:,1],yhat_r,label="Regresión de Ridge (L2)",c="darkcyan")
plt.legend(loc="best")