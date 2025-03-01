# MAQUINAS DE BOLTZMAN RESTRINGIDAS
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 10:20:23 2023

@author: Imanol
"""

# Importar las librerias ------------------------------------------------------
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable

# Importar el dataset
movies = pd.read_csv("ml-1m/movies.dat", sep= '::', header= None, 
                     engine='python', encoding='latin-1')
users = pd.read_csv("ml-1m/users.dat", sep= '::', header= None, 
                     engine='python', encoding='latin-1')
ratings = pd.read_csv("ml-1m/ratings.dat", sep= '::', header= None, 
                     engine='python', encoding='latin-1')

# Preparar el conjunto de entrenamiento y el conjunto de testing
training_set = pd.read_csv("ml-100k/u1.base", sep= '\t', header= None)
training_set = np.array(training_set, dtype= "int")
test_set = pd.read_csv("ml-100k/u1.test", sep= '\t', header= None)
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

# Convertir las valoraciones a valores binarios 1 (Me gusta) o 0 (No me gusta)
training_set[training_set == 0] = -1
training_set[training_set == 1] = 0
training_set[training_set == 2] = 0    
training_set[training_set >= 3] = 1

training_set[test_set == 0] = -1
training_set[test_set == 1] = 0
training_set[test_set == 2] = 0    
training_set[test_set >= 3] = 1

# Crear la arquitectura de la Red Neuronal (Modelo Probabilistico Gráfico)
class RBM():
    def __init__(self, nv, nh):
        self.W = torch.randn(nh, nv) # Inicia un tensor de tamaño nh x nv
        self.a = torch.randn(1, nh) # Vector bidimensional (lote y sesgo) 
        self.b = torch.randn(1, nv) # Vector bidimensional (lote y sesgo)
        
    def sample_h(self, x): # x = mini_batch_size x nv
        wx = torch.mm(x, self.W.t()) # mini_batch_size x nh
        activation = wx+self.a.expand_as(wx)
        p_h_given_v = torch.sigmoid(activation)
        return p_h_given_v, torch.bernoulli(p_h_given_v)
    
    def sample_v(self, y): # y = mini_batch_size x nh
        wy = torch.mm(y, self.W) # mini_batch_size x nv
        activation = wy+self.b.expand_as(wy)
        p_v_given_h = torch.sigmoid(activation)
        return p_v_given_h, torch.bernoulli(p_v_given_h)
    
    def train(self, v0, vk, ph0, phk):
        self.W += (torch.mm(v0.t(), ph0) - torch.mm(vk.t(), phk)).t()
        self.b += torch.sum(v0-vk, 0)
        self.a += torch.sum(ph0-phk, 0)
        
nv = len(training_set[0]) # nodos visibles (nº peliculas)
nh = 100 # 100 características para clasificar peliculas
batch_size = 100

rbm = RBM(nv, nh)

# Entrenar la RBM con 
nb_epoch = 10
for epoch in range(1, nb_epoch+1):
    training_loss = 0
    s = 0.
    for id_user in range(0, nb_users - batch_size, batch_size):
        vk = training_set[id_user:id_user+batch_size]
        v0 = training_set[id_user:id_user+batch_size]
        ph0, _ = rbm.sample_h(v0)
        for k in range(10):
            _, hk = rbm.sample_h(vk)
            _, vk = rbm.sample_v(hk)
            vk[v0 < 0] = v0[v0 < 0]
        phk, _ = rbm.sample_h(vk)
        rbm.train(v0, vk, ph0, phk)
        training_loss += torch.mean(torch.abs(v0[v0>=0]-vk[v0>=0])) # Distancia
        #Promedio 
        #training_loss += np.sqrt(torch.mean((v0[v0>=0] - vk[v0>=0])**2)) #RMSE
        s += 1.
    print("epoch: " + str(epoch) + ", loss: "+str(training_loss/s))
            
# Testear la RBM          
testing_loss = 0
s = 0.
for id_user in range(nb_users):
    v = training_set[id_user:id_user+1]
    vt = test_set[id_user:id_user+1]
    if len(vt[vt>=0]) > 0:
        _, h = rbm.sample_h(v)
        _, v = rbm.sample_v(h)
        testing_loss += torch.mean(torch.abs(vt[vt>=0] - v[vt>=0])) # Distancia
        # Promedio 
        #testing_loss += np.sqrt(torch.mean((vt[vt>=0] - v[vt>=0])**2)) # RMSE
        s += 1.
        print("Testing Loss: " + str(testing_loss/s))
              