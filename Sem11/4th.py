import cv2 as cv
import numpy as np
from sklearn.cluster import KMeans

img = cv.imread('image1.png')

k_vet = [8,64,512]

for k in k_vet:
    X = []
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            X.append(img[i][j])
    X = np.array(X)

    kmeans = KMeans(n_clusters=k).fit(X)
    centroids = kmeans.cluster_centers_

    print(k,centroids,"\n\n",set(kmeans.labels_),"\n\n",kmeans.labels_,"\n\n",(kmeans.labels_).shape)

    for it,l in enumerate(kmeans.labels_):
        X[it]= np.round(centroids[l])

    aux = 0
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            img[i][j] = X[aux]
            aux+=1

    cv.imshow("img", img)
    cv.waitKey(0)