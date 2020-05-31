# -*- coding: utf-8 -*-
"""
Created on Fri May 15 09:33:12 2020

@author: López Lazareno Diego Alberto 
"""
#%% Regresión lineal
# Parte 2. Regresión lineal múltiple y regresión polinómica

#%% Regresión lineal múltiple
# Se importan las librerías necesarias 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from mpl_toolkits import mplot3d
plt.style.use("ggplot")

# Lectura de datos
data=pd.read_csv("../Data/data_2d.csv")
# Bias
data["ones"]=1
# Variables independientes
x=np.array(data[["ones","x1","x2"]])
# Variable dependiente
y=np.array(data.iloc[:,2])

# Visualización 
fig=plt.figure()
ax=fig.add_subplot(111,projection="3d")
ax.scatter(x[:,1],x[:,1],y)

# Regresión lineal múltiple: forma matricial
# Coeficientes
w=np.linalg.solve(np.dot(x.T,x),np.dot(x.T,y))

# Predicción
yhat=np.dot(x,w)

# R^2 del modelo
ssr=np.sum((y-yhat)**2)
sst=np.sum((y-y.mean())**2)
r_2=1-ssr/sst
print("El R^2 del modelo es",r_2)

#%% Regresión polinómica

# lectura de datos ()
datan=pd.read_csv("../Data/data_poly.csv")
x=np.array(datan.iloc[:,0])
y=np.array(datan.iloc[:,1])

# Visualización de los datos
plt.figure(figsize=(10,5))
plt.xlabel("x")
plt.ylabel("y")
plt.scatter(x,y)

# Se procede a realizar una regresión polinómica
# Se añade el bias y los términos de x al cuadrado 
datan["ones"]=1
datan["x^2"]=datan["x"]**2
# Variable independiente
x=np.array(datan[["ones","x","x^2"]])
# Variable dependiente
y=np.array(datan["y"])

# Regresión polinómica: forma matricial
# Coeficientes
w=np.linalg.solve(np.dot(x.T,x),np.dot(x.T,y))

# Predicción
yhat=np.dot(x,w)

# Visualización de la curva ajustada
plt.figure(figsize=(10,5))
plt.xlabel("x")
plt.ylabel("y")
# Se ordenan las "x" y las "y" para que la curva se grafique bien
plt.plot(sorted(datan["x"]),sorted(yhat),c="black")
plt.scatter(datan["x"],datan["y"])

# R^2 del modelo
ssr=np.sum((y-yhat)**2)
sst=np.sum((y-y.mean())**2)
r_2=1-ssr/sst
print("El R^2 del modelo es",r_2)