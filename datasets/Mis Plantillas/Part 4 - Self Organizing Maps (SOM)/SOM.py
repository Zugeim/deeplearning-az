# Self Organizing Map
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 13:48:58 2023

@author: Imanol
"""

# Importar librerias
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importar el dataset
dataset = pd.read_csv("Credit_Card_Applications.csv")
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Escalado de características
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range= (0, 1))
X = sc.fit_transform(X)

# Entrenar el SOM
from minisom import MiniSom
som = MiniSom(x= 10, y= 10, input_len= 15, sigma= 1.0, learning_rate= 0.5)
som.random_weights_init(X)
som.train_random(data= X, num_iteration= 100)

# Visualizar los resutados
from pylab import bone, pcolor, colorbar, plot, show
bone()
pcolor(som.distance_map().T)
colorbar()
markers = ['o', 's']
colors = ['r', 'g']
for i, x in enumerate(X):
    w = som.winner(x)
    plot(w[0]+0.5, w[1]+0.5,
         markers[y[i]], 
         markeredgecolor= colors[y[i]],
         markerfacecolor= 'None',
         markersize= 10, markeredgewidth= 2)
show()

# Encontrar los fraudes
mappings = som.win_map(X)
frauds = np.concatenate((mappings[(6,4)], mappings[(8,3)], mappings[(7,3)], mappings[(8,7)]), axis= 0 ) # Coger todos los cuadraditos blancos
frauds = sc.inverse_transform(frauds)
