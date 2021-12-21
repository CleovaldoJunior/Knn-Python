import pandas as pd
from sklearn.model_selection import train_test_split


# Essa função faz justamente o cálculo da distância entre 2 linhas, ponto a ponto
# Retornando a raiz quadrada da soma das distâncias entre cada ponto
def euclidian_distance(linha1, linha2):
    distancia = 0
    for i in range(len(linha1)):
        distancia += (linha1[i] - linha2[i]) ** 2
    return distancia ** (1 / 2)


# Essa função vai retornar as N menores distâncias entre cada linha do teste
# com cada linha do treino, de forma que as distâncias são calculadas usando
# a função de distância euclidiana
def treino(teste, treino_x, treino_y, n):
    vizinhos_total = []
    for teste_atual in teste:
        distancias = []
        i = 0
        for treino_atual in treino_x:  # Para cada linha do teste percorre todas as linha do treino
            d_atual = euclidian_distance(teste_atual, treino_atual)  # Calcula a distância euclidiana
            distancias.append([list(treino_atual), d_atual, treino_y[i]])
            i += 1
        distancias.sort(key=lambda tup: tup[1])  # Ordena as distâncias de forma crescente
        vizinhos = distancias[:n]  # Pega as n menores distâncias
        vizinhos_total.append(vizinhos[0])
    return vizinhos_total


# Essa função basicamente faz o predict das menores distâncias
# se o Y da menor distância for igual ao Y da linha então tem-se mais um acerto
def predict(teste_x, teste_y, treino_x, treino_y, n):
    v = treino(teste_x, treino_x, treino_y, n)
    acc = 0
    for i in range(len(v)):
        if v[i][2] == teste_y[i]:
            acc += 1
    print(acc / len(teste_x) * 100)


# Usei a biblioteca pandas para carregar o data set
coluna = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
iris = pd.read_csv('iris.data', header=None, names=coluna)
classe_iris = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}
iris['species_num'] = [classe_iris[i] for i in iris.species]
X = iris.drop(['species', 'species_num'], axis=1)
y = iris.species_num

# Usei o train_test_split apenas para separar de forma mais fácil o treino do teste no dataset
treino_x, teste_x, treino_y, teste_y = train_test_split(X, y, test_size=0.5, random_state=42)
predict(teste_x=teste_x.values, teste_y=teste_y.values, treino_x=treino_x.values, treino_y=treino_y.values, n=1)
