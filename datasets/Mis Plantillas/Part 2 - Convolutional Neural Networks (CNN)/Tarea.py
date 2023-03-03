# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 13:11:23 2023

@author: Imanol
"""

# Importación de librerias
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam

batch_size = 512 # 64
target_size = (128, 128)

# Carga de imagenes -----------------------------------------------------------
train_datagen = ImageDataGenerator(rescale= 1./255,
                                   shear_range= 0.2,
                                   zoom_range= 0.2,
                                   horizontal_flip= True)

test_datagen = ImageDataGenerator(rescale= 1./255)


training_dataset = train_datagen.flow_from_directory('dataset/training_set',
                                                     target_size= target_size, 
                                                     batch_size= batch_size, 
                                                     class_mode= 'binary')

testing_dataset = test_datagen.flow_from_directory('dataset/test_set',
                                                   target_size= target_size, 
                                                   batch_size= batch_size, 
                                                   class_mode= 'binary')


# Creación de la CNN ----------------------------------------------------------

## Inicializar la CNN
classifier = Sequential()

## Convolución
classifier.add(Conv2D(filters= 32, kernel_size= (3, 3), input_shape= (*target_size, 3), activation= "relu"))
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
classifier.add(Dropout(rate= 0.3)) 
classifier.add(Dense(units= 128, activation= 'relu')) 
classifier.add(Dropout(rate= 0.5))
classifier.add(Dense(units= 1, activation= 'sigmoid'))

## Compilar la CNN
adam = Adam(learning_rate= 0.0003)
#adam = 'adam'
classifier.compile(optimizer= adam, loss= 'binary_crossentropy', metrics= 'accuracy')
    

# Ejecución de la CNN ---------------------------------------------------------

classifier = classifier.fit(training_dataset,
               steps_per_epoch= training_dataset.n / batch_size,
               epochs=64, 
               validation_data= testing_dataset,
               validation_steps= testing_dataset.n / batch_size,
               workers= 12,
               verbose= 2)


# Graficar Resultados ---------------------------------------------------------

import matplotlib.pyplot as plt

acc      = classifier.history[     'accuracy']
val_acc  = classifier.history[ 'val_accuracy']
loss     = classifier.history[         'loss']
val_loss = classifier.history[     'val_loss']

epochs   = range(1,len(acc)+1,1)

plt.plot ( epochs,     acc, 'r--', label= 'Training acc')
plt.plot ( epochs, val_acc,  'b', label= 'Validation acc')
plt.title ('Training and validation accuracy')
plt.ylabel('acc')
plt.xlabel('epochs')

plt.legend()
plt.figure()

plt.plot ( epochs,     loss, 'r--' )
plt.plot ( epochs, val_loss,  'b' )
plt.title ('Training and validation loss'   )
plt.ylabel('loss')
plt.xlabel('epochs')

history_dict = classifier.history
#print(history_dict.keys())

plt.legend()
plt.figure()
print(acc)
print(max(val_acc))

