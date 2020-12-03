#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# In[2]:


def sigmoid(x):
    return 1 / (1 + np.exp(-x))
# sigmoid derivatives to adjust synaptic weights
def sigmoid_derivative(x):
    return x * (1 - x)


# In[10]:


# Dados de entrada do sustema
training_inputs = np.array([[0,0,1],
                            [1,1,1],
                            [1,0,1],
                            [0,1,1]])
training_outputs = np.array([[0,1,1,0]]).T


# In[18]:


np.random.seed(1)
a=training_inputs[1].shape
b=training_outputs.shape
b


# In[5]:


# initialize weights randomly with mean 0 to create weight matrix, synaptic weights
synaptic_weights = 2 * np.random.random((3,1)) - 1

print('Random starting synaptic weights: ')
print(synaptic_weights)


# In[6]:


a=300
lista=[]
lista2=[]
# Itera a vezes
for iteration in range(a):

    # definindo dados de entrada
    input_layer = training_inputs
    # normalizando o produto dos valores de entradas pelo peso das sinapses
    outputs = sigmoid(np.dot(input_layer, synaptic_weights))

    # erro do sistema
    error =  outputs-  training_outputs

    # mmultiplica o erro pela curva do valor dos elementos de saida
    adjustments = error * sigmoid_derivative(outputs)

    # atualiza pesos
    synaptic_weights += np.dot(input_layer.T, adjustments)
    
    #calcula o erro 
    lista.append(sum(error))
    lista2.append(sum(adjustments))
#     lista2.append(np.exp(-sum(error)))


# In[7]:


print('pesos sinapses apos o treino: ')
print(synaptic_weights)

print("saida apos o treino:")
print(outputs.round(3))


# In[8]:


lista
x=[i for i in range(a)]


# In[9]:


import matplotlib.pyplot as plt
plt.plot(x, lista), plt.grid(True)
plt.plot(x,lista2), plt.grid(True)


# In[ ]:




