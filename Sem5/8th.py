import cv2
import numpy as np

img = cv2.imread('imagens/Fig0314(a)(100-dollars).png',0)

def transformacao(img):
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if 0 <= img[i][j] <= (254/8)*4:
                pass
            else:
                img[i][j] = 255
    return img

t1 = transformacao(img)

# Save edited images.
cv2.imshow('t1', t1)
cv2.waitKey(0)
