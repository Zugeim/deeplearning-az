# -*- coding: utf-8 -*-
"""
Created on Wed Mar  1 12:27:08 2023

@author: Imanol
"""

# TAREA

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# Codificar datos categóricos
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

def build_classifier(optimizer= "rmsprop", drop1= 0., drop2= 0.):
    
    classifier = Sequential()
    
    classifier.add(Dense(units= 6, kernel_initializer= "uniform", activation= "relu", input_dim= 11))  
    classifier.add(Dropout(rate= drop1))
    
    classifier.add(Dense(units= 6, kernel_initializer= "uniform", activation= "relu"))
    classifier.add(Dropout(rate= drop2))

    
    classifier.add(Dense(units= 1, kernel_initializer= "uniform", activation= "sigmoid"))
        


    classifier.compile(optimizer= optimizer, loss= "binary_crossentropy", metrics= ["accuracy"])

    return classifier

classifier = KerasClassifier(build_fn= build_classifier)


parameters = {
    'batch_size': [25],
    'epochs': [500],
    'optimizer': ['rmsprop', 'SGD', 'Adadelta'],
    'drop1': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
    'drop2': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    }

grid_search = GridSearchCV(estimator= classifier, 
                           param_grid= parameters,
                           scoring= 'accuracy',
                           cv= 10,
                           n_jobs= -1)

grid_search = grid_search.fit(X_train, y_train)
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_
best = best_parameters
best['accuracy'] = best_accuracy