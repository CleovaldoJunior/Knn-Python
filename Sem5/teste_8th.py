import numpy as np
import cv2
# Read the image in greyscale
img = cv2.imread('imagens/Fig0314(a)(100-dollars).png',0)

#itera sobre cada píxel da imagem e adiciona o binário do mesmo no vetor lst
lst = []
for i in range(img.shape[0]):
    for j in range(img.shape[1]):
         lst.append(np.binary_repr(img[i][j] ,width=8)) # width = no. of bits

#Para extrair o plano de bit eu só iterei sobre cada string do binário e guardei os bits
#correspondentes do plano de bit em uma lista
#Multipliquei com 2^(n-1) e dei reshape para reconstruir a imagem
p = [1,2,4,8,16,32,64,128]
for j in range(len(p)):
    print(j, p[j])
    img_final = (np.array([int(i[j]) for i in lst],dtype = np.uint8) * p[7-j]).reshape(img.shape[0],img.shape[1])
    cv2.imshow('a',img_final)
    cv2.waitKey(0)