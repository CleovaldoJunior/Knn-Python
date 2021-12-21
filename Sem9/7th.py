import cv2
import numpy as np

filter = [[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]

img = cv2.imread('imagens/equacoes.png')
img = 255 - img
img_media = cv2.blur(img,(3,3))
img_sub = img - img_media
saida = img + (1*img_sub)

_, img_gradient_t = cv2.threshold(saida, 100, 255, cv2.THRESH_BINARY)
cv2.imwrite("imagem.png", img_gradient_t)
cv2.waitKey(0)