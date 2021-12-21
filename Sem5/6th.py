import cv2
import numpy as np

img = cv2.imread('imagens/Fig0310(b)(washed_out_pollen_image).png',0)
img2 = cv2.imread('imagens/Fig0310(b)(washed_out_pollen_image).png',0)

#Primeira tranformação
def transformacao(img):
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i][j] < 64:
                img[i][j] *= .5
            elif 64 <= img[i][j] <= 192:
                img[i][j] = (img[i][j]*1.5)-65
            elif img[i][j] > 192:
                img[i][j] = (img[i][j]*.5)+128
    return img

#Segunda transformação
def transformacao_2(img):
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i][j] < 107:
                img[i][j] = 0
            elif img[i][j] >= 107:
                img[i][j] = 255
    return img

t1 = transformacao(img)
t2 = transformacao_2(img2)

cv2.imshow('t1', t1)
cv2.imshow('t2', t2)
cv2.waitKey(0)
