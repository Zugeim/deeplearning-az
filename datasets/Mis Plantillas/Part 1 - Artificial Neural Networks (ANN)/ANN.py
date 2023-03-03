# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

# REDES NEURONALES ARTIFICIALES

# Instalar Theano
# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

# Instalar Tensorflow y Keras
# conda install -c conda-forge keras

# Parte 1 - Pre Procesado de Datos ---------------------------------------------------------

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# Codificar datos categóricos
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
ct = ColumnTransformer(
    [('one_hot_encoder', # Un nombre de la transformación
      OneHotEncoder(categories='auto'), # La clase a la que transformar
      [1]) # Las columnas a transformar
     ],remainder='passthrough')
X = np.array(ct.fit_transform(X), dtype=np.float)
X = X[:, 1:]

# Division del data set en conjunto de entrenamiento y conjunto de testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Escalado de Variables
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)



# Parte 2 - Construir la RNA ---------------------------------------------------------------

# Importar Keras y librerias adicionales
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

# Inicializar la RNA
classifier = Sequential()

# Añadir las capas de entrada y primera capa oculta
classifier.add(Dense(units= 6, # Nodos de la capa oculta (Generalmente (nEntrada+nSalida)/2)
                     kernel_initializer= "uniform", # Como inicializar los pesos
                     activation= "relu", # Función de activación de la capa oculta
                     input_dim= 11)) # Nodos de entrada (Las variables de entrada) 
               
# Añadir la segunda capa oculta
classifier.add(Dense(units= 6, # Nodos de la capa oculta (Generalmente (nEntrada+nSalida)/2)
                     kernel_initializer= "uniform", # Como inicializar los pesos
                     activation= "relu")) # Función de activación de la capa oculta

# Añadir la capa de salida
classifier.add(Dense(units= 1, # Nodos de la capa de salida
                     kernel_initializer= "uniform", # Como inicializar los pesos
                     activation= "sigmoid")) # Función de activación de la capa

# Compilar la RNA
classifier.compile(optimizer= "adam", # Método de optimización de la función de coste
                   loss= "binary_crossentropy", # Función de perdidas (coste)
                   metrics= ["accuracy"])

# Ajustar el clasificador al Conjunto de Testing
classifier.fit(X_train, y_train, batch_size= 10, epochs= 100)



# Parte 3 - Evaluar el modelo y calcular predicciones finales ------------------------------

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = y_pred > 0.5

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

(cm[0][0] + cm[1][1]) / (cm.sum())

# TAREA
X_new = sc.transform([[0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])
y_new_pred = classifier.predict(X_new) > 0.5



# Parte 4 - Evaluar, Mejorar y Ajustar la RNA ----------------------------------------------

# Evaluar la RNA
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score

def build_classifier():
    
    classifier = Sequential()
    classifier.add(Dense(units= 6, kernel_initializer= "uniform", activation= "relu", input_dim= 11))  
    classifier.add(Dense(units= 6, kernel_initializer= "uniform", activation= "relu"))  
    classifier.add(Dense(units= 1, kernel_initializer= "uniform", activation= "sigmoid"))
        
    classifier.compile(optimizer= "adam", loss= "binary_crossentropy", metrics= ["accuracy"])

    return classifier

classifier = KerasClassifier(build_fn= build_classifier, batch_size= 10, epochs= 100)
accuracies = cross_val_score(estimator= classifier, X = X_train, y= y_train, cv= 10, n_jobs= -1, verbose= 1)

mean = accuracies.mean()
variance = accuracies.std()

print(mean)
print(variance)

# Mejorar la RNA

## Regularización de Dropout para evitar el overfitting
def build_classifier_DO():
    
    classifier = Sequential()
    
    classifier.add(Dense(units= 6, kernel_initializer= "uniform", activation= "relu", input_dim= 11))  
    # Añadir capa DropOut
    classifier.add(Dropout(rate= 0.1))
    
    classifier.add(Dense(units= 6, kernel_initializer= "uniform", activation= "relu"))
    # Añadir capa DropOut
    classifier.add(Dropout(rate= 0.1))
    
    classifier.add(Dense(units= 1, kernel_initializer= "uniform", activation= "sigmoid"))
        


    classifier.compile(optimizer= "adam", loss= "binary_crossentropy", metrics= ["accuracy"])

    return classifier

classifier = KerasClassifier(build_fn= build_classifier_DO, batch_size= 10, epochs= 100)
accuracies = cross_val_score(estimator= classifier, X = X_train, y= y_train, cv= 10, n_jobs= -1, verbose= 1)

mean = accuracies.mean()
variance = accuracies.std()

print(mean)
print(variance)

# Ajustar la RNA
from sklearn.model_selection import GridSearchCV

def build_classifier_final(optimizer= "adam"):
    
    classifier = Sequential()
    
    classifier.add(Dense(units= 6, kernel_initializer= "uniform", activation= "relu", input_dim= 11))  
    # Añadir capa DropOut
    classifier.add(Dropout(rate= 0.1))
    
    classifier.add(Dense(units= 6, kernel_initializer= "uniform", activation= "relu"))
    # Añadir capa DropOut
    classifier.add(Dropout(rate= 0.1))
    
    classifier.add(Dense(units= 1, kernel_initializer= "uniform", activation= "sigmoid"))
        


    classifier.compile(optimizer= optimizer, loss= "binary_crossentropy", metrics= ["accuracy"])

    return classifier

classifier = KerasClassifier(build_fn= build_classifier_final)

parameters = {
    'batch_size': [25, 32],
    'epochs': [100, 500],
    'optimizer': ['adam', 'rmsprop']
    }

grid_search = GridSearchCV(estimator= classifier, 
                           param_grid= parameters,
                           scoring= 'accuracy',
                           cv= 10)
grid_search = grid_search.fit(X_train, y_train)
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_

# TAREA +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def build_classifier(optimizer= "rmsprop", act1= "relu", act2= "relu", act3= "simoid"):
    
    classifier = Sequential()
    
    classifier.add(Dense(units= 6, kernel_initializer= "uniform", activation= act1, input_dim= 11))  
   
    
    classifier.add(Dense(units= 6, kernel_initializer= "uniform", activation= act2))

    
    classifier.add(Dense(units= 1, kernel_initializer= "uniform", activation= act3))
        


    classifier.compile(optimizer= optimizer, loss= "binary_crossentropy", metrics= ["accuracy"])

    return classifier

classifier = KerasClassifier(build_fn= build_classifier)

parameters = {
    'batch_size': [32],
    'epochs': [250],
    'optimizer': ['rmsprop', 'SGD', 'Adadelta'],
    'act1': ['relu', 'sigmoid', 'softmax', 'softplus', 'softsign', 'tanh', 'selu', 'eslu', 'exponential'],
    'act2': ['relu', 'sigmoid', 'softmax', 'softplus', 'softsign', 'tanh', 'selu', 'eslu', 'exponential'],
    'act3': ['relu', 'sigmoid', 'softmax', 'softplus', 'softsign', 'tanh', 'selu', 'eslu', 'exponential']
    }

grid_search = GridSearchCV(estimator= classifier, 
                           param_grid= parameters,
                           scoring= 'accuracy',
                           cv= 10,
                           n_jobs= -1)

grid_search = grid_search.fit(X_train, y_train)
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_
