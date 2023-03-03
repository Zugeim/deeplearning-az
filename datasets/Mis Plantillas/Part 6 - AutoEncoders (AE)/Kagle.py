# -*- coding: utf-8 -*-
"""
Created on Thu Mar  2 18:58:33 2023

@author: Imanol
"""

# Importar las librerias 
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable

# Importar el dataset
training_set = pd.read_csv("Kaggle/train_v2.csv")
training_set = np.array(training_set, dtype= "int")
test_set = pd.read_csv("Kaggle/test_v2.csv")
test_set = np.array(training_set, dtype= "int")

# Obtener el número de usuarios y de peliculas
nb_users = int(max(max(training_set[:, 0]), max(test_set[:,0])))
nb_movies = int(max(max(training_set[:, 1]), max(test_set[:,1])))

# Convertir los datos en un array X[u, i] con usuarios u en fila y 
# peliculas i en columna
def convert(data):
   new_data = []
   for id_user in range(1, nb_users+1):
       id_movies = data[:, 1][data[:,0] == id_user]
       id_ratings = data[:, 2][data[:,0] == id_user]
       ratings = np.zeros(nb_movies)
       ratings[id_movies-1] = id_ratings
       new_data.append(list(ratings))
   return new_data

training_set = convert(training_set)
test_set = convert(test_set)

# Convertir los datos a tensores de Torch
training_set = torch.FloatTensor(training_set)
test_set = torch.FloatTensor(test_set)

# Crear la arquitectura de la Red Neuronal
class SAE(nn.Module):
    def __init__(self, ):
        super(SAE, self).__init__()
        self.fc1 = nn.Linear(nb_movies, 30)
        self.fc2 = nn.Linear(30, 15)
        self.fc3 = nn.Linear(15, 30)
        self.fc4 = nn.Linear(30, nb_movies)
        self.activation = nn.Sigmoid()
        
    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        x = self.fc4(x)
        return x
  
sae = SAE()
criterion = nn.MSELoss()
optimizer = optim.RMSprop(sae.parameters(), lr = 0.003, weight_decay = 0.5)

# Entrenar el SAE
nb_epoch = 200
for epoch in range(1, nb_epoch+1):
    train_loss = 0
    s = 0.
    for id_user in range(nb_users):
        inpt = Variable(training_set[id_user]).unsqueeze(0) # Para adecuar el vector
        target = inpt.clone()
        if torch.sum(target.data > 0) > 0:
            outpt = sae.forward(inpt)
            target.require_grad = False
            outpt[target == 0] = 0
            loss = criterion(outpt, target)
            mean_correcto = nb_movies / float(torch.sum(target.data > 0)+1e-10)
            loss.backward()
            train_loss += np.sqrt(loss.data*mean_correcto)
            s += 1.
            optimizer.step()
    print("Epoch: "+str(epoch)+", Loss: "+str(train_loss/s))
    
# Evaluar el conjunto de test en nuestro SAE
test_loss = 0
s = 0.
for id_user in range(nb_users):
    inpt = Variable(training_set[id_user]).unsqueeze(0) # Para adecuar el vector
    target = Variable(test_set[id_user]).unsqueeze(0)
    if torch.sum(target.data > 0) > 0:
        outpt = sae.forward(inpt)
        target.require_grad = False
        outpt[target == 0] = 0
        loss = criterion(outpt, target)
        mean_correcto = nb_movies / float(torch.sum(target.data > 0)+1e-10)
        test_loss += np.sqrt(loss.data*mean_correcto)
        s += 1.
        optimizer.step()
print("Test Loss: "+str(test_loss/s))