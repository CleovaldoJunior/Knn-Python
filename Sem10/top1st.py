import numpy as np

#Classe base
import pandas as pd


class Camada:
    def __init__(self):
        self.input = None
        self.output = None

    #Calcula o Y de uma camada dado um X
    def forward_propagation(self, input):
        raise NotImplementedError

    #Calcula dE/dX dado um dE/dY
    def backward_propagation(self, output_error, learning_rate):
        raise NotImplementedError

class Network:
    def __init__(self):
        self.camadas = []
        self.perda = None
        self.perda_principal = None

    #Adiciona camada à rede
    def add(self, camada):
        self.camadas.append(camada)

    #Seta uma perda
    def usar(self, perda_param, perda_principal_param):
        self.perda = perda_param
        self.perda_principal = perda_principal_param

    #Pred output dado um input
    def predict(self, input):
        #Amostra dimensão primeira
        amostra = len(input)
        resultado = []

        #Roda a rede para todas as amostras
        for i in range(amostra):
            #Forward Propagation
            output = input[i]
            for camada in self.camadas:
                output = camada.forward_propagation(output)
            resultado.append(output)

        return resultado

    #Treina a Rede
    def fit(self, x_train, y_train, epocas, taxa_de_aprendizagem):
        #Amostra dimensão primeira
        amostras = len(x_train)

        #Treinamento
        for i in range(epocas):
            err = 0
            for j in range(amostras):
                #Forward Propagation
                output = x_train[j]
                for layer in self.camadas:
                    output = layer.forward_propagation(output)

                #Guarda a perda
                # compute loss (for display purpose only)
                err += self.perda(y_train[j], output)

                #Backward Propagation
                erro = self.perda_principal(y_train[j], output)
                for camada in reversed(self.camadas):
                    erro = camada.backward_propagation(erro, taxa_de_aprendizagem)

            #Calcula a média de todos os erros
            err /= amostras
            #print('epoch %d/%d   error=%f' % (i+1, epocas, err))

class FCLayer(Camada):

    #Recebe o número de neuronios do input e output
    def __init__(self, tamanho_input, tamanho_output):
        self.pesos = np.random.rand(tamanho_input, tamanho_output) - 0.5
        self.bias = np.random.rand(1, tamanho_output) - 0.5

    def forward_propagation(self, input):
        self.input = input
        self.output = np.dot(self.input, self.pesos) + self.bias
        return self.output

    #Calcuyla dE/dW, dE/dB dado um erro de output como dE/dY, e retorna dE/dX
    def backward_propagation(self, erro_output, taxa_de_aprendizagem):
        input_error = np.dot(erro_output, self.pesos.T)
        erro_pesos = np.dot(self.input.T, erro_output)

        #Atualiza os parâmetros
        self.pesos -= taxa_de_aprendizagem * erro_pesos
        self.bias -= taxa_de_aprendizagem * erro_output
        return input_error

class ActivationLayer(Camada):
    def __init__(self, ativacao_param, ativacao_principal_param):
        self.ativacao = ativacao_param
        self.ativacao_principal = ativacao_principal_param

    #Retorna o input ativado
    def forward_propagation(self, input):
        self.input = input
        self.output = self.ativacao(self.input)
        return self.output

    #Retorna dE/dX dado um erro de ouput dE/dY
    def backward_propagation(self, erro_output, taxa_de_aprendizagem):
        return self.ativacao_principal(self.input) * erro_output

#Função de Perda
def mse(y_true, y_pred):
    return np.mean(np.power(y_true-y_pred, 2))

def mse_prime(y_true, y_pred):
    return 2*(y_pred-y_true)/y_true.size

#Função de ativação
def tanh(x):
    return np.tanh(x)

def tanh_prime(x):
    return 1-np.tanh(x)**2

# training data
coluna = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
iris = pd.read_csv('iris.data', header=None, names=coluna)
classe_iris = {'Iris-setosa':1, 'Iris-versicolor':0, 'Iris-virginica':0}
iris['species_num'] = [classe_iris[i] for i in iris.species]
x = iris.drop(['species', 'species_num'], axis=1).values
y = iris.species_num.values
x_treino = []
y_treino = []
for i in range(len(x)):
    x_treino.append(np.array(np.array(x[i])))
for j in range(len(y)):
    y_treino.append(np.array(np.array(y[j])))

# network
net = Network()
net.add(FCLayer(0, 0))
net.add(ActivationLayer(tanh, tanh_prime))

# train
net.usar(mse, mse_prime)
net.fit(x_treino, y_treino, epocas=1000, taxa_de_aprendizagem=0.1)