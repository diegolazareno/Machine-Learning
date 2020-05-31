# -*- coding: utf-8 -*-
"""
Created on Thu May 14 12:56:29 2020

@author: López Lazareno Diego Alberto
"""
#%% Regresión lineal 
# Parte 1. Regresión lineal en 1 dimensión 

# Se importan las librerías necesarias
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use("ggplot")

# Lectura de datos (github de Lazyprogrammer)
data=pd.read_csv("../Data/data_1d.csv")
# Variable independiente
x=np.array(data.iloc[:,0])
# Variable dependiente
y=np.array(data.iloc[:,1])
# Samples
n=len(x)

# Visualización
plt.figure(figsize=(10,5))
plt.xlabel("x")
plt.ylabel("y")
plt.scatter(x,y,c="darkcyan")

# Se aplica la fórmula derivada para la regresión lineal en 1-D
# El denominador es común para ambas expresiones
denom=np.dot(x,x)/n-(np.mean(x))**2
# Pendiente
m=np.dot(x,y)/n-np.mean(x)*np.mean(y)
m=m/denom
# Bias 
b=np.mean(y)*np.dot(x,x)/n-np.mean(x)*np.dot(x,y)/n
b=b/denom

# Visualización de la recta que mejor ajusta
yhat=m*x+b
plt.figure(figsize=(10,5))
plt.xlabel("x")
plt.ylabel("y")
plt.scatter(x,y,c="darkcyan")
plt.plot(x,yhat)

# R^2
# Suma de los residuales al cuadrado
ssr=np.sum((y-yhat)**2)
# Suma del total al cuadrado
sst=np.sum((y-np.mean(y))**2)
r_2=1-(ssr/sst)
print("El R^2 es", r_2)