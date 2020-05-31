# -*- coding: utf-8 -*-
"""
Created on Mon May 18 11:30:33 2020

@author: López Lazareno Diego Alberto
"""

#%% Regresión logística 
# Parte 1. la función sigmoide (neurona)

# Se importan las librerías necesarias
import numpy as np

# Creación de datos
N=100 # Samples
D=2 # Dimensión de los samples
X=np.random.randn(N,D)
ones=np.array([[1] for i in range(N)])
Xb=np.concatenate((ones,X),axis=1) # Variable independiente
w=np.random.randn(D+1) # Pesos (w0,w1,w2)
z=np.dot(Xb,w) # Producto punto (suma ponderada)

# Sigmoide (neurona)
def sigmoide(Z):
    return 1/(1+np.exp(-Z)) # Forma vectorial

# Output de la neurona para cada sample (entre 0 y 1)
output=sigmoide(z)
output