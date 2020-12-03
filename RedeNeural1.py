import numpy as np

class NeuralNetwork():
    
    def __init__(self):
        #seed do numpy para numeros aleatorios, caso um
        np.random.seed(1)

        # criando os valores das sinapses em uma matriz 3,1
        # os valores vão de -1 ate 1 
        self.peso_sinapse = 2 * np.random.random((4, 1)) - 1

    def sigmoid(self, x):
        """ pega os valores das somas do input e normaliza esses valores atraves de uma sigmoid
            ou seja os valores das somas dos inputs estarão entre 0 e 1 """
        return 1 / (1 + np.exp(-x))

    def derivada_sigmoid(self, x):
        """ 
        A derivação da sigmoid para calcular o ajuste dos erros """
        return x * (1 - x)

    def train(self, treino_entrada, treino_saida, treino_iteracao):
        """
        treinamos o modelo atraves de tentativa e erro, ajustando o peso das sinapses para resultados melhores
        """
        for iteracao in range(treino_iteracao):
            # passa o processo de treinamento na redeneural 
            output = self.pensamento(treino_entrada)

            # Calcula o erro do processo
            error = treino_saida - output

            # Multiplica o erro do input e os gradientes da função sigmoid com objetivo de ajustar o erro
            adjustments = np.dot(treino_entrada.T, error * self.derivada_sigmoid(output))

            # ajusta o peso das sinapses
            self.peso_sinapse += adjustments

    def pensamento(self, inputs):
        """
        Passa as entradas atraves da rede neural para oder fazer as analises
        """
        
        inputs = inputs.astype(float)
        output = self.sigmoid(np.dot(inputs, self.peso_sinapse))
        return output


#if __name__ == "__main__":

# # inicializa a rede neural
neural_network = NeuralNetwork()

print("Inicio aleatorio dos pesos das sinapses")
print(neural_network.peso_sinapse)

# o treino consiste de q exemplos  com com quatro entradas de entrada e uma de saida,
# cada exemplo é constituido de 4 valores, 
treino_entrada = np.array([[0,0,1,1],
                            [1,1,1,0],
                            [1,0,1,0],
                            [0,1,1,10]])

treino_saida = np.array([[0,1,1,0]]).T

# Train the neural network
neural_network.train(treino_entrada, treino_saida, 10000)

print("Synaptic weights after training: ")
print(neural_network.peso_sinapse)

A = str(input("Input 1: "))
B = str(input("Input 2: "))
C = str(input("Input 3: "))
D = str(input("Input 4: "))
print("New situation: input data = ", A, B, C,D)
print("Output data: ")

teste=neural_network.pensamento(np.array([A, B, C,D]))
print(teste)
# print(neural_network.pensamento(np.array([A, B, C, D]))
