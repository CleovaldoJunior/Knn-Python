import cv2

import numpy as np

img = cv2.imread('imagens/Fig0334(a)(hubble-original).png',0)

img_filtro_media = cv2.medianBlur(img,15)
cv2.imshow("Img_mediana",img_filtro_media)
cv2.waitKey(0)
_, img_t = cv2.threshold(img, 64, 255, cv2.THRESH_BINARY)
cv2.imshow("Img_limiar_64",img_t)
cv2.waitKey(0)
img_multiplicada = img_t * img
cv2.imshow("Img_multiplicada",img_multiplicada)
cv2.waitKey(0)