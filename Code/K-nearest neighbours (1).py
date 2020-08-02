# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 13:12:12 2020

@author: López Lazareno Diego Alberto 
"""

#%% K-Nearest Neighbours 
# Parte 1. KNN (MNIST dataset)

# Librerías 
import pandas as pd
import numpy as np
from sortedcontainers import SortedList

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
    

if __name__=="__main__":
    # Lectura de archivos 
    df=pd.read_csv("../Data/train.csv")
    data=df.as_matrix()
    np.random.shuffle(data)
    X=data[:2000,1:]/255.0 
    Y=data[:2000,0]
    
    # Datos de entrenamiento y prueba
    xtrain,xtest=X[:1000],X[1000:]
    ytrain,ytest=Y[:1000],Y[1000:]
   
    # Resultados con distintos k-vecinos más cercanos
    train_accu={}
    test_accu={}
    for k in range(1,5):
        knn=KNN(k)
        knn.fit(xtrain,ytrain)
        train_accu[str(k)+" nearest neighbours"]="Train accuracy: "+ str(knn.score(xtrain,ytrain))
        test_accu[str(k)+" nearest neighbours"]="Test accuracy: "+ str(knn.score(xtest,ytest))