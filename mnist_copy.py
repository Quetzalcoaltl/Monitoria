#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf  # deep learning library. Tensors are just multi-dimensional arrays
#                         biblioteca pra aprendizado de Deep Learning. Tensos são vetores multi dimensionais


# In[2]:



mnist = tf.keras.datasets.mnist  # mnist is a dataset of 28x28 images of handwritten digits and their labels
# mnist é um arquivo com varios numeros escritos a mão, é um data set contendo varias numeros digitados e como eles são tabelados, ou seja, se vc pegar o valor dele, tem tambem o valor da reposta em outro lugar
(x_train, y_train),(x_test, y_test) = mnist.load_data()  # unpacks images to x_train/x_test and labels to y_train/y_test
# retira os dados que queremos, estamos simplesmente pegando os dados, temos os dados de treino e alguns para testarmos, bem simples, 

x_train = tf.keras.utils.normalize(x_train, axis=1)  # normaliza meus dados entre 0 e 1, para que fique mais rapido, 
print(x_test[1])
x_test = tf.keras.utils.normalize(x_test, axis=1)  # normaliza meus dados entre 0 e 1, não é fundamentalmente necessario ḿas facilita nossas predições
print(x_test[1])


# In[ ]:



model = tf.keras.models.Sequential()  # modelo basixo de inicio de IA, a basic feed-forward model
model.add(tf.keras.layers.Flatten())  # takes our 28x28 and makes it 1x784
# pega a imagem, do tipo 28 po 28 e transforma em uma de 1 por 784
# model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))  # a simple fully-connected layer, 128 units, relu activation
model.add(tf.keras.layers.Dense(128, activation='relu'))
# aqui a gente diz quantidade de neuronios e selecionamos a ativação
# por que 128? porque é um numero interessante de neuronios, não é pequeno que não funcione e não é grnde demais que vai atrapalhar seu modelo. Esse numero, 128 é 2^7 e é mais facil paa o pc calcular

# muito acima disso tem problemas de overfiting, que é quando o sistema aprende mais do que deve porque ele começa a aprender com os erros do sistema. Eu aconselho vcs a pegarem isso aqui e ficarem alterando os dados, para ver como isso muda o sistema
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))  # a simple fully-connected layer, 128 units, relu activation
# camada oculta de 128 nos com ativação relu, podemos pensar que isso realizamos a cada 
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))  # our output layer. 10 units for 10 classes. Softmax for probability distribution

# model.add(tf.keras.layers.Dense(10, activation='relu'))


# In[ ]:



model.compile(optimizer='adam',  # Good default optimizer to start with
              loss='sparse_categorical_crossentropy',  # how will we calculate our "error." Neural network aims to minimize loss.
              metrics=['accuracy'])  # what to track


# In[ ]:



model.fit(x_train, y_train, epochs=3)  # train the model

val_loss, val_acc = model.evaluate(x_test, y_test)  # evaluate the out of sample data with model
print(val_loss)  # model's loss (error)
print(val_acc)  # model's accuracy


# In[ ]:


# model.save('epic_num_reader.model')


# In[ ]:


predictions = model.predict(x_test)
print(predictions)


# In[ ]:


import numpy as np
print(np.argmax(predictions[0]))


# In[ ]:


import matplotlib.pyplot as plt
plt.imshow(x_train[0])
plt.show()


# In[ ]:


# x_test[1]
y_train[0]


# In[ ]:




