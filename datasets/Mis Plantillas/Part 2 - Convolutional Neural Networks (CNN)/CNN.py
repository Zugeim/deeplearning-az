# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 09:51:26 2023

@author: Imanol
"""

# REDES NEURONALES CONVOLUCIONALES

# Parte 1 - Construir el modelo CNN ---------------------------------------------------------

## Importar las librerias y paquetes
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

## Inicializar la CNN
classifier = Sequential()

## Paso 1 - Convolución
classifier.add(Conv2D(filters= 32, # 32 mapas de carectaristicas
                             kernel_size= (3, 3), # Tamaño de cada mapa 3x3
                             input_shape= (64, 64, 3), # Tamaño de las imagenes de entrada
                             activation= "relu"))

## Paso 2 - Max Pooling
classifier.add(MaxPooling2D(pool_size= (2,2)))

# Segunda capa de convolución y max pooling
classifier.add(Conv2D(filters= 32, kernel_size= (3, 3), activation= "relu"))
classifier.add(MaxPooling2D(pool_size= (2,2)))

## Paso 3 - Flattering
classifier.add(Flatten())

## Paso 4 - Full Conection
classifier.add(Dense(units= 128, activation= 'relu'))
classifier.add(Dense(units= 1, activation= 'sigmoid'))

## Compilar la CNN
classifier.compile(optimizer= 'adam', loss= 'binary_crossentropy', metrics= 'accuracy')

# Parte 2 - Ajustar la CNN a las imágenes para entrenar ---------------------------------------------------------
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale= 1./255,
                                   shear_range= 0.2,
                                   zoom_range= 0.2,
                                   horizontal_flip= True)

test_datagen = ImageDataGenerator(rescale= 1./255)

batch_size = 32

training_dataset = train_datagen.flow_from_directory('dataset/training_set',
                                                     target_size= (64,64), 
                                                     batch_size= batch_size, 
                                                     class_mode= 'binary')

testing_dataset = test_datagen.flow_from_directory('dataset/test_set',
                                                   target_size= (64,64), 
                                                   batch_size= batch_size, 
                                                   class_mode= 'binary')

classifier.fit(training_dataset,
               steps_per_epoch= training_dataset.n / batch_size,
               epochs=25,
               validation_data= testing_dataset,
               validation_steps= 2000,
               workers= 10,
               verbose= 1)


# Parte 3 - Creando nueva predicción ---------------------------------------------------------

import numpy as np
from keras.utils import load_img, img_to_array

test_image = load_img('dataset/single_prediction/cat_or_dog_1.jpg', target_size= (64,64))
test_image = img_to_array(test_image)
test_image = np.expand_dims(test_image, axis= 0)

result = classifier.predict(test_image)

#training_dataset.class_indices

if result[0][0] == 1:
    prediction = 'dog'
else:
    prediction = 'cat'
    
print(prediction)


# TAREA ----------------------------------------------------------------------------------------
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from keras.layers import Dropout

## Inicializar la CNN
classifier = Sequential()

## Convolución
classifier.add(Conv2D(filters= 32, kernel_size= (3, 3), input_shape= (64, 64, 3), activation= "relu"))
classifier.add(Conv2D(filters= 32, kernel_size= (3, 3), activation= "relu"))

## Max Pooling
classifier.add(MaxPooling2D(pool_size= (2,2)))

# Segunda capa de convolución y max pooling
classifier.add(Conv2D(filters= 64, kernel_size= (3, 3), activation= "relu"))
classifier.add(Conv2D(filters= 64, kernel_size= (3, 3), activation= "relu"))

## Max Pooling
classifier.add(MaxPooling2D(pool_size= (2,2)))

## Paso 3 - Flattering
classifier.add(Flatten())

## Paso 4 - Full Conection
classifier.add(Dense(units= 128, activation= 'relu'))
classifier.add(Dropout(rate= 0.1))
classifier.add(Dense(units= 128, activation= 'relu'))
classifier.add(Dropout(rate= 0.1))
classifier.add(Dense(units= 1, activation= 'sigmoid'))

## Compilar la CNN
classifier.compile(optimizer= 'adam', loss= 'binary_crossentropy', metrics= 'accuracy')
    
classifier.fit(training_dataset,
               steps_per_epoch= training_dataset.n / batch_size,
               epochs=25,
               validation_data= testing_dataset,
               validation_steps= testing_dataset.n / batch_size,
               workers= 11,
               verbose= 1)