import numpy as np
from sklearn import datasets
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt

iris = datasets.load_iris()
X = iris.data
Y = iris.target

k = 9
kmeans = KMeans(n_clusters=k).fit(X)
k_grupos = [[0,0,0] for _ in range(k)]


pred = kmeans.predict(X)

for i, t in enumerate(Y):
    p = pred[i]
    k_grupos[p][t] += 1

print(k_grupos)
for it, i in enumerate(k_grupos):
    print(it)
    plt.bar([1,2,3],i,0.5)
    plt.xticks([1,2,3],['setosa','versicolor','virginica'])
    plt.title('k '+str(k)+' Cluster Nº '+str(it+1))
    plt.show()
