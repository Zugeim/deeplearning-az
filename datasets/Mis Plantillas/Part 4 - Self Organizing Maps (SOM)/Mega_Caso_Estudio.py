# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 10:56:07 2023

@author: Imanol
"""

# Parte 1 - Identificar los fraudes con un SOM ----------------------------------------------------------------------------------------------

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
som = MiniSom(x= 20, y= 20, input_len= 15, sigma= 1.0, learning_rate= 0.5)
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
frauds = np.concatenate((mappings[(3,6)], mappings[(4,6)], mappings[(5,3)], mappings[(8,2)]), axis= 0 ) # Coger todos los cuadraditos blancos
frauds = sc.inverse_transform(frauds)

# Parte 2 - Trasladar el modelo de Deep Learning de no supervisado a supervisado ------------------------------------------------------------

# Crear matriz de caracteristicas
customers = dataset.iloc[:, 1:-1].values

# Crear la variable dependiente
is_fraud = np.zeros(len(dataset))
for i in range(len(dataset)):
    if dataset.iloc[i, 0] in frauds:
        is_fraud[i] = 1
   
# Escalado de Variables
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
customers = sc.fit_transform(customers)


# Construir la RNA 

# Importar Keras y librerias adicionales
from keras.models import Sequential
from keras.layers import Dense

# Inicializar la RNA
classifier = Sequential()

# Añadir las capas de entrada y primera capa oculta
classifier.add(Dense(units= 2, kernel_initializer= "uniform", activation= "relu", input_dim= 14)) 
               
# Añadir la capa de salida
classifier.add(Dense(units= 1, kernel_initializer= "uniform", activation= "sigmoid"))

# Compilar la RNA
classifier.compile(optimizer= "adam", loss= "binary_crossentropy", metrics= ["accuracy"])

# Ajustar el clasificador al Conjunto de Testing
classifier.fit(customers, is_fraud, batch_size= 1, epochs= 2)


# Evaluar el modelo y calcular predicciones finales 

# Predicting the Test set results
y_pred = classifier.predict(customers)
y_pred = np.concatenate((dataset.iloc[:, 0:1].values, y_pred), axis= 1)
y_pred = y_pred[(-y_pred[:,1]).argsort()] # Lo pongo en negativo para que sea descendente el orden


