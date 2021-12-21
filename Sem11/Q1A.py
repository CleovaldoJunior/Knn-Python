import numpy as np
from sklearn.cluster import KMeans
from math import dist
from sklearn import datasets

iris = datasets.load_iris()
data = iris.data
target = iris.target

k_grupos = [[[],[],[]] for _ in range(10)]

for k in [3,6,9]:
    for j in range(10):
        kmeans = KMeans(n_clusters=k, max_iter=j+1).fit(data)

        centros = kmeans.cluster_centers_
        p = kmeans.predict(data)

        for i,g in enumerate(p):
            centro_prox = centros[g]
            d = dist(data[i], centro_prox)
            k_grupos[j][int((k/3)-1)].append(d)


for ite, i in enumerate(k_grupos):
    for it,j in enumerate(i):
        print((it+1)*3,"Media",sum(j)/len(j))
        print((it+1)*3,"Desvio:",np.std(j))
    print(ite,"=======")




# print(matrix)
# c = 0
# for linha in matrix.T:
#     c +=3
#     print('Media para K=' + str(c) + ':', np.sum(linha)/10 )
#     print('Desvio Padrão:', np.std(linha))
#     print('=============================')