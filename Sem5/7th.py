import cv2
import numpy as np

img = cv2.imread('imagens/Fig0312(a)(kidney).png',0)
img2 = cv2.imread('imagens/Fig0312(a)(kidney).png',0)
def transformacao(img):
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if 127 <= img[i][j] <= 127*1.7:
                img[i][j] = 153
            else:
                img[i][j] = 25
    return img

def transformacao_2(img):
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if 127 <= img[i][j] <= 127*1.7:
                img[i][j] = 204
    return img

t1 = transformacao(img)
t2 = transformacao_2(img2)

cv2.imshow('t1', t1)
cv2.imshow('t2', t2)
cv2.waitKey(0)
