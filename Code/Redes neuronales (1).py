# -*- coding: utf-8 -*-
"""
Created on Fri May 29 12:33:34 2020

@author: López Lazareno Diego Alberto
"""

#%% Redes Neuronales
# Parte 1. Softmax

# Se importan las librerías necesarias
import numpy as np

# Output (activación) en la última capa de la neurona (1 sample)
a=np.random.randn(5) # 5 clases
expa=np.exp(a) # Exponenciar el vector
prob=expa/expa.sum() # Probabilidades de pertenecer a la K-ésima clase
prob.sum() # Si sumamos las probabilidades deben dar 1

# Mismo ejercicio anterior pero con N samples
a_=np.random.randn(100,5) # 100 samples y 5 clases
exp_=np.exp(a_)
prob_=exp_/exp_.sum(axis=1,keepdims=True) # Probabilidades de que cada sample pertenezca a la K-ésima clase
# El argumento keepdims permite que la operación efectuada se guarde en una matriz en vez de en un vector
prob_.sum(axis=1) # La suma en cada fila da 1