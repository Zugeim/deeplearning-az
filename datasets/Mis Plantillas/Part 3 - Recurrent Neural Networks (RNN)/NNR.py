# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 13:17:14 2023

@author: Imanol
"""

# REDES NEURONALES RECURRENTES

# Parte 1 - Procesado de datos ------------------------------------------------

# Importación de las librerias
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importar el dataset de entrenamiento
dataset_train = pd.read_csv("Google_Stock_Price_Train.csv")
training_set = dataset_train.iloc[:, 1:2].values

# Escalado de caracteristicas
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range= (0, 1))
training_set_scaled = sc.fit_transform(training_set)

# Crear una estructura de datos con 60 timesteps y 1 salida
X_train = []
y_train = []

for i in range(60, dataset_train.shape[0]):
    X_train.append(training_set_scaled[i-60:i, 0])
    y_train.append(training_set_scaled[i, 0])
    
X_train, y_train = np.array(X_train), np.array(y_train)

# Redimensión de los datos
## Por si queremos meter más datos como datos de otras empresas u otros datos
## de la misma empresa deberiamos cambiar el 1
X_train = np.reshape(X_train, 
                     (X_train.shape[0], X_train.shape[1], 1)) # Nueva estructura



# Parte 2 - Construcción de la RNR --------------------------------------------
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout

# Inicialización de módelos
regressor = Sequential()

# Añadir la primera capa de LSTM y la regularización por Dropout
regressor.add(LSTM(units= 50, # Tiene que ser un valor alto
                   return_sequences= True, # Tru si quiero meter más capas
                   input_shape=  (X_train.shape[1], 1) )) # El número de columnas y cuantas variables regresoras
regressor.add(Dropout(0.2))

# Añadir la segunda capa de LSTM y la regularización por Dropout
regressor.add(LSTM(units= 50, return_sequences= True)) 
regressor.add(Dropout(0.2))

# Añadir la tercera capa de LSTM y la regularización por Dropout
regressor.add(LSTM(units= 50, return_sequences= True)) 
regressor.add(Dropout(0.2))

# Añadir la cuarta capa de LSTM y la regularización por Dropout
regressor.add(LSTM(units= 50)) 
regressor.add(Dropout(0.2))

# Añadir la capa densa de salida
regressor.add(Dense(units= 1)) 

# Compilar la RNR
regressor.compile(optimizer= 'Nadam', loss= 'mean_squared_error' )

# Ajustar la RNR a nuestro conjunto de entrenamiento
regressor.fit(x= X_train, y= y_train,
              epochs= 100,
              batch_size= 32,
              verbose= 2,
              workers= 12)


# Parte 3 - Ajustar las predicciones y visualizar los resultados --------------

# Obtener el valor real de las acciones de Enero de 2017
dataset_test = pd.read_csv("Google_Stock_Price_Test.csv")
real_stock_price = dataset_test.iloc[:, 1:2].values

# Obtener la predicción de la acción con la RNR para Enero de 2017 
dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis= 0)
inputs = dataset_total[len(dataset_total)-len(dataset_test)-60: ].values
inputs = inputs.reshape(-1, 1)
inputs = sc.transform(inputs)

X_test = []
for i in range(60, inputs.shape[0]):
    X_test.append(inputs[i-60:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1) )

predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

# Visualizar los resultados
plt.plot(real_stock_price, color= 'red', label= 'Precio Real de la Accion de Google')
plt.plot(predicted_stock_price, color= 'blue', label= 'Precio Predicho de la Accion de Google')
plt.title('Prediccion con una RNR del valor de las acciones de Google')
plt.xlabel('Fecha')
plt.ylabel('Precio de la accion de Google')
plt.legend()
plt.show()