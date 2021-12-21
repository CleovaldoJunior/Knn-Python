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

#Carrega o Dataset
dataset_url = 'wine.data'
data = pd.read_csv(dataset_url,names=['y','1','2','3','4','5','6','7','8','9','10','11','12','13'])
x = data.drop(['y'],axis=1)
y = data.y

#Classifica com a ultima feature
scores_total = []
for i in range(100):
    knn = KNeighborsClassifier(n_neighbors=1, p=2)
    X_train, X_test, y_train, y_test = train_test_split(x, y,test_size=0.5,shuffle=True)
    knn.fit(X_train, y_train)
    score = knn.score(X_test, y_test)
    scores_total.append(score)

#Dropo a ultima feature a classifico novamente
x = data.drop(['13'],axis=1)
scores_drop = []
for i in range(100):
    knn = KNeighborsClassifier(n_neighbors=1, p=2)
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.5, shuffle=True)
    knn.fit(X_train, y_train)
    score = knn.score(X_test, y_test)
    scores_drop.append(score)

#Calculo a diferença entre as duas
dife = []
for i in range(100):
    dif = round(scores_drop[i]-scores_total[i],3)
    dife.append(dif)
    print(dif)

print("Confiança Diferença:", confidence_interval(dife))
print("Confiança com ultima feature:",confidence_interval(scores_total))
print("Confiança sem ultima feature:",confidence_interval(scores_drop))

f1_2 = plt.hist(dife, bins=25)
plt.title("Histograma das Diferenças")
plt.show()