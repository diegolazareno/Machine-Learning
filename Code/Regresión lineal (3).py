# -*- coding: utf-8 -*-
"""
Created on Fri May 15 17:00:14 2020

@author: López Lazareno Diego Alberto 
"""
#%% Regresión lineal
# Parte 3. Regresión lineal múltiple (presión sistólica)

# Se importan las librerías necesarias
import pandas as pd
import numpy as np

# Lectura de archivos 
data=pd.read_excel("../Data/mlr02.xls")
data.columns=["Presión sistólica","Edad","Peso (libras)"]
data["ones"]=1

# Computar distintas regresiones y evaluar sus R^2
# Variable dependiente (presión sistólica)
y=data.iloc[:,0]
# Variable independiente (edad)
x_edad=data[["ones","Edad"]]
# Variable independiente (peso)
x_peso=data[["ones","Peso (libras)"]]
# Variables independientes (edad y peso)
x=data[["ones","Edad","Peso (libras)"]]

# Regresión lineal
def lin_reg(x,y):
    x,y=np.array(x),np.array(y)
    w=np.linalg.solve(np.dot(x.T,x),np.dot(x.T,y))
    yhat=np.dot(x,w)
    ssr=y-yhat
    sst=y-y.mean()
    r_2=1-np.dot(ssr,ssr)/np.dot(sst,sst)
    return w,r_2

# 1er modelo 
w_1,r_1=lin_reg(x_edad,y)
# 2do modelo
w_2,r_2=lin_reg(x_peso,y)
# 3er modelo
w_3,r_3=lin_reg(x,y)

print("R^2 del 1er modelo",r_1)
print("R^2 del 2do modelo",r_2)
print("R^2 del 3er modelo",r_3) # Mejor modelo (regresión lineal múltiple)

#%% Regresión lineal

# Se añade un componente aleatorio al primer modelo para reevaluar su R^2
data["componente aleatorio"]=np.random.randn(len(data))
new_x=data[["ones","Edad","componente aleatorio"]]

# Nuevo modelo
w,new_r2=lin_reg(new_x,y)
print("R^2",new_r2) # El R^2 mejora un 1% cuando se añade el c. aleatorio

# La mejora (auque baja) en el R^2 se da porque el componente aleatorio sí
# tiene correlación con la variable dependiente, pues, al ser una muestra,
# no se cumple la correlación esperada (0) 