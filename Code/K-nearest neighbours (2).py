# -*- coding: utf-8 -*-
"""
Created on Sat Aug  1 10:55:06 2020

@author: USUARIO
"""

#%% K-Nearest Neighbours 
# Parte 2. KNN (XOR y donut)

# Librerías 
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from sortedcontainers import SortedList

#%% Generación de datos
# Problema XOR
def get_data_1():
    # Se generan 200 puntos de dos dimensiones
    X=np.zeros((200,2))
    # Nube (X~[0.5,1],Y~[0.5,1])
    X[:50]=np.random.random((50,2))/2+0.5 
    # Nube (X~[0,1],Y~[0,1])
    X[50:100]=np.random.random((50,2))/2
    # Nube (X~[0,0.5],Y~[0.5,1])
    X[100:150]=np.random.random((50,2))/2+np.array([[0,0.5]])
    # Nube (X~[0.5,1],Y~[0,0.5])
    X[150:]=np.random.random((50,2))/2+np.array([[0.5,0]])
    Y=np.array([0]*100+[1]*100)
    return X,Y

# Problema del donut
def get_data_2():
    # Se generan 100 puntos
    N=100
    # Radio interno de 5 unidades
    R_inner=5
    # Radio externo de 10 unidades
    R_outer=10
    
    # Coordenadas polares para el círculo interno
    R1=np.random.randn(N)+R_inner # Radio de 5 unidades con ruido gaussiano
    theta=2*np.pi*np.random.random(N) # Ángulo uniformemente distribuido ~[0,2pi]
    # Conversión a coordenadas rectangulares
    X_inner=np.concatenate([[R1*np.cos(theta)],[R1*np.sin(theta)]]).T
    
    # Coordenadas polares para el círculo externo
    R2=np.random.randn(N)+R_outer # Radio de 10 unidades con ruido gaussiano
    theta=2*np.pi*np.random.random(N) # Ángulo uniformemente distribuido ~[0,2pi]
    # Conversión a coordenadas rectangulares
    X_outer=np.concatenate([[R2*np.cos(theta)],[R2*np.sin(theta)]]).T
    
    # Se concatenan ambos círculos para graficarlos al mismo tiempo
    X=np.concatenate((X_inner,X_outer))
    Y=np.array([0]*N+[1]*N)
    return X,Y    

#%% KNN
# K-Nearest Neighbours (objeto)
class KNN(object):
    # Inicializar la función con k-vecinos más cercanos
    def __init__(self,k):
        self.k=k
    
    # Ajuste de los datos de entrenamiento 
    def fit(self,X,Y):
        self.X=X
        self.Y=Y
    
    # Predicción
    def predict(self,X):
        # Clase a la que pertenece cada punto 
        y=[]
        
        # Se obtiene la distancia de cada punto del conjunto de prueba con cada punto del conjunto de entrenamiento 
        for i,x in enumerate(X):
            sl=SortedList()
            for j,xt in enumerate(self.X):
                diff=x-xt
                d=diff.dot(diff)
                if len(sl)<self.k:
                    sl.add((d,self.Y[j]))
                else:
                    if d<sl[-1][0]:
                        del sl[-1]
                        sl.add((d,self.Y[j]))
            
            # Clases propuestas para cada punto
            labels=[]
            for i in range(len(sl)):
                labels.append(sl[i][1])
            
            # Clase con mayor número de repeticiones
            y.append(pd.value_counts(labels).idxmax())
            
        return y
    
    # Exactitud del modelo
    def score(self,X,Y):
        P=self.predict(X)
        return np.mean(P==Y)
       
#%%
if __name__=="__main__":
    # Problema XOR
    X,Y=get_data_1()
    plt.figure()
    plt.scatter(X[:,0],X[:,1],c=Y,s=100,alpha=0.5)
    # Modelo con 3-vecinos más cercanos
    knn=KNN(3)
    knn.fit(X,Y)
    # Se obtiene una exactitud alta con 3-vecinos más cercanos
    print("Accuracy score for XOR problem:",knn.score(X,Y))
    
    # Problema del donut
    X,Y=get_data_2()
    plt.figure()
    plt.scatter(X[:,0],X[:,1],c=Y,s=100,alpha=0.5)
    # Modelo con 3-vecinos más cercanos
    knn=KNN(3)
    knn.fit(X,Y)
    # Se obtiene una exactitud alta con 3-vecinos más cercanos
    print("Accuracy score for donut problem:",knn.score(X,Y))