# coding=utf-8
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

#Converte os elementos do dataset que são Str
#Para numérico
def numerico(dataset):
    #Percorre as colunas
    for i in range(len(dataset.columns)):
        coluna = dataset.iloc[:,i].values
        set_coluna = list(set(coluna))
        d = {}
        c = 0
        #Para cada elemento da coluna atual
        for ele in set_coluna:
            #Se o elemento for string
            if isinstance(ele, str):
                #O elemento é trocado pelo seu index
                d[ele] = c
            c += 1
        #Basicamente trocando a coluna antiga pela nova
        for j in range(len(coluna)):
            if isinstance(coluna[j], str):
                coluna[j] = d[coluna[j]]
    return dataset

#Transforma o target do dataset em binário
def binario(dataset):
    coluna = dataset.iloc[:,]
    #Peguei a moda da feature, no caso
    #o valor que mais aparece na coluna
    moda = dataset.mode().values[0]
    #Percorri a coluna transformando tudo que fosse
    #maior ou igual que a moda em 1
    #e tudo que fosse menor em 0
    for ele in range(len(coluna)):
        if coluna[ele] >= moda:
            coluna[ele] = 1
        else:
            coluna[ele] = 0
    return dataset

atts = ["school","sex","age","address","famsize","Pstatus","Medu","Fedu","Mjob","Fjob",
        "reason","guardian","traveltime","studytime","failures","schoolsup","famsup","paid",
        "activities","nursery","higher","internet","romantic","famrel","freetime","goout",
        "Dalc","Walc","health","absences","G1","G2","y"]

students = pd.read_csv('student-mat2.csv', header=None, names=atts)
X = students.drop("y", axis=1)
y = students.y

X = numerico(X.copy())
y = binario(y.copy())

scores = []
knn = KNeighborsClassifier(n_neighbors=1,p=2)
for i in range(100):
    X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.5, shuffle=True)
    knn.fit(X_train, y_train)
    score = knn.score(X_test, y_test)
    scores.append(round(score,3))
print("  Média do score:", sum(scores)/len(scores))
print(confidence_interval(scores))

