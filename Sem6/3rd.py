import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
from statistics import NormalDist

#Função que usei para calcular o intervalo de confiança com 95%
def confidence_interval(data, confidence=0.95):
  dist = NormalDist.from_samples(data)
  z = NormalDist().inv_cdf((1 + confidence) / 2.)
  h = dist.stdev * z / ((len(data) - 1) ** .5)
  return dist.mean - h, dist.mean + h

def numerico(dataset):
    for i in range(len(dataset.columns)):
        coluna = dataset.iloc[:,i].values
        set_coluna = list(set(coluna))
        d = {}
        c = 0
        for ele in set_coluna:
            if isinstance(ele, str):
                d[ele] = c
            else:
                d[ele] = ele
            c += 1
        nova_coluna = []
        for j in range(len(coluna)):
            if isinstance(coluna[j], str):
                coluna[j] = d[coluna[j]]
    return dataset

def binario(dataset):
    coluna = dataset.iloc[:,]
    moda = dataset.mode().values[0]
    for ele in range(len(coluna)):
        if coluna[ele] >= moda:
            coluna[ele] = 1
        else:
            coluna[ele] = 0
    return dataset

atts = ["x1","x2","x3","x4","x5","x6","y"]

car = pd.read_csv('car.data', header=None, names=atts)
X = car.drop("y", axis=1)
y = car.y

print(np.array(X.values))
print(np.array(y.values))

X = numerico(X.copy())
y = binario(y.copy())

print(np.array(X.values))
print(np.array(y.values))

