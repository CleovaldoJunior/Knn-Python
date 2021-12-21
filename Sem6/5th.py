from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
from statistics import NormalDist
from statistics import stdev

#Função que usei para calcular o intervalo de confiança com 95%
def confidence_interval(data, confidence=0.95):
  dist = NormalDist.from_samples(data)
  z = NormalDist().inv_cdf((1 + confidence) / 2.)
  h = dist.stdev * z / ((len(data) - 1) ** .5)
  return dist.mean - h, dist.mean + h

#Troca cada atributo pelo seu respectivo
#no intervalo [0,1]
def intervalo(dataset):
    #Percorre as colunas
    for key in dataset.keys():
        #Pega o menor valor da coluna
        menor = min(dataset[key])
        #Pega o maior valor da coluna
        maior = max(dataset[key])
        #Percorre cada elemento da coluna aplicando
        #a formula de intervalo
        for i in range(len(dataset[key])):
            ele = dataset[key][i]
            ele_novo = (ele - menor)/(maior-menor)
            dataset[key][i] = ele_novo
    print(dataset)
    return dataset

def padronizacao(dataset):
    for key in dataset.keys():
        media = sum(dataset[key])/len(dataset[key])
        desvio_padrao = stdev(dataset[key])
        for i in range(len(dataset[key])):
            ele = dataset[key][i]
            ele_novo = (ele - media)/desvio_padrao
            dataset[key][i] = ele_novo
    print(dataset)
    return dataset


atts = ["y","x1","x2","x3","x4","x5","x6","x7","x8","x9","x10","x11","x12","x13"]

hungarian = pd.read_csv('wine.csv', header=None, names=atts)
X = hungarian.drop("y", axis=1)
y = hungarian.y

#X = padronizacao(X.copy())

scores = []
knn = KNeighborsClassifier(n_neighbors=1,p=2)
for i in range(100):
    X_train, X_test, y_train, y_test = train_test_split(X, y,train_size=0.5)
    knn.fit(X_train, y_train)
    score = knn.score(X_test, y_test)
    scores.append(round(score,3))
print("   Média do score:", sum(scores)/len(scores))
print(confidence_interval(scores))

