import time

import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn_extra.cluster import KMedoids
from sklearn.neighbors import KNeighborsClassifier

iris = datasets.load_iris()
X = iris.data
Y = iris.target

k_vet = [9,18,27,45,72]

for it, k in enumerate(k_vet):
    print("============================ k =",k,"===============================")
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.5, shuffle=True, stratify=Y)
    kmedoids = KMedoids(n_clusters=k, random_state=0).fit(X_train)

    print("Labels Treino:",y_train,"\n")
    print("Labels Kmedoids:",kmedoids.labels_,"\n")

    knn = KNeighborsClassifier(n_neighbors=1,p=2)
    x_centroids_treino = kmedoids.cluster_centers_
    y_centroids_treino = []
    for it, fc in enumerate(x_centroids_treino):
        for itx, fx in enumerate(X_train):
            if (fc == fx).all():
                y_centroids_treino.append(y_train[itx])
    d_pred_acc_class = {0:0,1:0,2:0}

    print("X do centroides:",x_centroids_treino,"\n","Y dos centroides:",y_centroids_treino)
    t = time.time()
    knn.fit(x_centroids_treino, y_centroids_treino)
    pred = knn.predict(X_test)
    print("Tempo:",time.time()-t)
    for it, f in enumerate(y_test):
        if f == pred[it]:
            d_pred_acc_class[f] += 1
    u, c = np.unique(y_test,return_counts=True)
    d_y_count_class_test = dict(zip(u,c))
    for i in d_y_count_class_test.keys():
        print("Acc:","classe:",i,"=",(d_pred_acc_class[i]/d_y_count_class_test[i])*100)



