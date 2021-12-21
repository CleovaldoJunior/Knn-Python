import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from statistics import NormalDist

#Função que usei para calcular o intervalo de confiança com 95%
def confidence_interval(data, confidence=0.95):
  dist = NormalDist.from_samples(data)
  z = NormalDist().inv_cdf((1 + confidence) / 2.)
  h = dist.stdev * z / ((len(data) - 1) ** .5)
  return dist.mean - h, dist.mean + h

#Carrego o Dataset
coluna = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
iris = pd.read_csv('iris.data', header=None, names=coluna)
classe_iris = {'Iris-setosa':0, 'Iris-versicolor':1, 'Iris-virginica':2}
iris['species_num'] = [classe_iris[i] for i in iris.species]
X = iris.drop(['species', 'species_num'], axis=1)
y = iris.species_num
scores = []

#Faço 100 holdouts para cada K de 1 a 16 e calculo o intervalo de confiança
for k in range(1,16):
    knn = KNeighborsClassifier(n_neighbors=k,p=2)
    for i in range(100):
        X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.5, shuffle=True)
        knn.fit(X_train, y_train)
        score = knn.score(X_test, y_test)
        scores.append(round(score,3))

    print("K =",k,"|",confidence_interval(scores))

