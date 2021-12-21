import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt

from statistics import NormalDist

#Função que usei para calcular o intervalo de confiança com 95%
def confidence_interval(data, confidence=0.95):
  dist = NormalDist.from_samples(data)
  z = NormalDist().inv_cdf((1 + confidence) / 2.)
  h = dist.stdev * z / ((len(data) - 1) ** .5)
  return dist.mean - h, dist.mean + h

#Apenasa prepara o dataset
def preparar_dataset(filename):
    df = pd.read_csv(filename)
    data = pd.DataFrame(df)
    target = data['class']
    data.drop('class', inplace=True, axis=1)
    n = data.shape[1]
    data = np.array(data)
    target = np.array(target)
    target = np.reshape(target, [len(target), 1])
    target = target.T[0]
    return data, target

x, y = preparar_dataset("Skin.data")

f_score_array = []
score_array = []
knn = KNeighborsClassifier(n_neighbors=1, p=2)
rkf = StratifiedKFold(n_splits=100) #Aqui que eu defini que seria estratificado e a quantidade de Folds
for train_index, test_index in rkf.split(x, y=y):
    X_train, X_test = x[train_index], x[test_index]
    y_train, y_test = y[train_index], y[test_index]
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    score = knn.score(X_test, y_test)
    f_score_array.append(f1_score(y_test, y_pred, average=None)) #Peguei o F-Score de cada iteração

f1_1class = []
f1_2class = []
for i in f_score_array:
    f1_1class.append(i[0])
    f1_2class.append(i[1])

print("Min:",min(f1_1class), "Max:",max(f1_1class),"Med:",sum(f1_1class)/len(f1_1class))
print("Min:",min(f1_2class), "Max:",max(f1_2class),"Med:",sum(f1_2class)/len(f1_2class))

f1_1 = plt.hist(f1_1class, bins=25)
plt.title("Histograma da Medida-F da Classe 1")
plt.show()
f1_2 = plt.hist(f1_2class, bins=25)
plt.title("Histograma da Medida-F da Classe 2")
plt.show()

print(confidence_interval(f1_1class))
print(confidence_interval(f1_2class))