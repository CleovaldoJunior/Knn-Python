import numpy as np
import cv2
# Read the image in greyscale
img = cv2.imread('imagens/Fig0314(a)(100-dollars).png',0)

lst = []
for i in range(img.shape[0]):
    for j in range(img.shape[1]):
         lst.append(np.binary_repr(img[i][j] ,width=8)) # width = no. of bits

#Peguei apenas os bits 128 e 64 e somei os mesmos para gerar a imagem combinada
img_128 = (np.array([int(i[0]) for i in lst],dtype = np.uint8) * 128).reshape(img.shape[0],img.shape[1])
img_64 = (np.array([int(i[1]) for i in lst],dtype = np.uint8) * 64).reshape(img.shape[0],img.shape[1])
img_final = img_128+img_64
cv2.imshow("Imagem top",img_final)
cv2.waitKey(0)