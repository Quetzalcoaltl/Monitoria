#!/usr/bin/env python
# coding: utf-8

# In[11]:


import math
import numpy as np
import os
import csv
import tensorflow as tf
# from tensorflow import keras.models
from keras.models import Sequential
from keras.layers import Dense


# In[13]:


arquivos_csv= os.listdir('./Dados_CSVs_2703')
print(arquivos_csv)
# index_arquivos=len(arquivos_csv)

# print(index_arquivos)


# In[1]:
# teste para escrever em arquivos documentos

# with open('teste.txt','w+') as file: 
#     for i in range(10):
#         file.write(str(i))
# a = [i for i in range(10)]


# In[5]:


contador=0 # contador para garantir que eu crio matrix3 igual a matrix somente uma vez no codigo 
# matrix3 = np.empty(shape=(0,0),dtype='object') 

# CRIAR UMA MATRIX DE SAIDA
# output = numero de colunas dela vai ser o numero de eventos que eu tenho, 



for a in arquivos_csv:
    obj= "./Dados_CSVs_2703/" + a
    aux=[] #ressetando minha lista que sera utilizada para armazenar os valores
    with open(obj, newline='') as file:
        reader = csv.reader(file)
        res = list(map(tuple, reader))
        for j2 in range(len(res)):
            aux.append(res[j2]) # colocando os valores dentro da lista vazia
            
    aux1=np.array(aux) # transformando minha lista em um array
    aji=len(aux1) #numero de linhas
    bji=len(aux1[0]) #numero de colunas
    matrix = np.empty(shape=(aji-1,bji),dtype='object')  # essa é a matriz que é alocada dinamicamente a cada interação da lista de CSVs
    h=(aji-1, index_arquivos)
    matrix2=np.zeros(h) # criando/reiniciando a matrix dinamica vazia de zeros base para a saida
    
#     print(len(matrix))
#     print(len(matrix2))

    for i in range(aji):
        
        if i == 0:
            print("\n cabeçalho: "+a) #pulando o cabeçalho para que a mtriz contenha apenas os valores reais
        else:
            vet=aux1[i]
#             print("entrou no loop") # verificação
            for j in range(bji):
               
                if vet[j] == 'False': # fazendo o ajuste dos valores booleanos
                    c=0
                elif vet[j] == 'True':
                    c=1
                else:
                    c = vet[j] # para alocar o 
                matrix2[i-1][arquivos_csv.index(a)]= 1 # matrix dinamica de saida
                matrix[i-1][j]= float(c) # normalizando os valores todos para float, garantindo que minha matriz só tem valores
    if contador: # checando o contador, se for é necessario criar a matriz, com as dimensões especidificadas
        contador +=1
#         print("não foi aqui")
        saida=np.concatenate((saida, matrix2))
        entrada=np.concatenate((entrada, matrix)) # concatenate coloca cada matriz uma em cima da outra
    else:
        contador +=1 # contador para garantir que eu somente aloco esses valores na primeira vez
#         print("foi aqui") 
        saida=matrix2
        entrada=matrix

print("\nQuantas vezes foi iterado meu sistema: ",contador)
index_csv=len(entrada[0])
print('Comprimento vetor de entrada: ',index_csv)


# In[18]:


# criar um algoritmo para randomizar os dados, pois os resultados são melhores dessa forma, ela pode aprender ja com uma boa distribuição
a=np.array(entrada) #(5,18)
b=np.array(saida) #(5,7)
print('matriz a:\n',a)
print('matriz b:\n',b)

c = np.c_[a.reshape(len(a), -1), b.reshape(len(b), -1)]
print('matriz c: \n', c)
np.random.shuffle(c)
print('matriz c randomizada: \n', c)
entrada3= c[:, :a.size//len(a)].reshape(a.shape)
saida3= c[:, a.size//len(a):].reshape(b.shape)
print('matriz a2:\n',entrada3)
print('matriz b2:\n',saida3)


# In[21]:


entrada_alt=np.matrix(entrada3).astype(float) 
entrada1= tf.keras.utils.normalize(entrada_alt, axis =1) #criando o array normalizado, no qual os valores não são maiores que 1


saida=np.array(saida3).astype(float) #(5,7)
# print("dimensões entrada:", entrada2.shape)
print("dimensões saida:", saida.shape)
neuronios=13 #(18+7)/2 = 12.5
camadas=2 #duas camadas pois o trabalho não é muuuito complexo


classificador = Sequential()
classificador.add(Dense(units=13, activation= 'relu', input_dim=index_csv)) #primeira camada oculta com 18 entradas
classificador.add(Dense(units=13, activation= 'relu')) #segunda camada oculta
# classificador.add(Dense(units=13, activation= 'relu')) # terceira camada oculta
classificador.add(Dense(units= index_arquivos, activation= 'sigmoid')) #saida com tamanho da saida(7 elementos)
classificador.compile(optimizer ='adam', loss='binary_crossentropy',metrics=['accuracy'])


epocas= 20 # quantidade de vezes que o sistema vai iterar sobre os elementos de teste


###################ATENÇÃO######################
###### COMENTAR UMA(1) DAS DUAS(2) PARTES ABAIXO, CASO CONTRARIO VAI RESULTAR EM ERRO
# In[22]:

#################### 1 ###########

print("\n resultados e treino com entradas normalizadas:\n")
hist=classificador.fit(entrada1,
                       saida,
                       validation_split=0.25,
                       verbose=1,                        
                       batch_size=10,
                       shuffle=True,
                       epochs=epocas)


# In[20]:

#################### 2 ###########
'''
print("\n resultados e treino com entradas normalizadas e randomizadas: \n")

entrada3=np.matrix(entrada3).astype(float) 
# entrada=entrada.astype(float)
entrada3= tf.keras.utils.normalize(entrada3, axis =1)
hist=classificador.fit(entrada3,
                       saida3.astype(float),
                       validation_split=0.25,
                       verbose=1,
                       batch_size=10,
                       epochs=epocas+10)
'''

# In[14]:


# https://keras.io/visualization/ 
#  visualização com matplotlib, acessando o conteudo do historico do treinamento feito com keras

import matplotlib.pyplot as plt

print(hist.history, '\n conteudo do dictionary')
preci=hist.history['accuracy']
print(preci)
perdas=hist.history['loss']
print(perdas)
b=[i for i in range(len(preci))]
plt.plot(b, preci), plt.grid(True)
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

plt.plot(b, perdas), plt.grid(True)
plt.ylabel('perdas')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()




##############outra forma de se visualizar os dados


# val_ac=hist.history['val_acc']

# print(preci)
# plt.plot(hist.history['acc'])

# plt.plot(hist.history['val_acc'])
# plt.title('Model accuracy')
# plt.ylabel('Accuracy')
# plt.xlabel('Epoch')
# plt.legend(['Train', 'Test'], loc='upper left')
# plt.show()
# print(aux.content())
# historia=classificador.history['acc']
# print(historia)




