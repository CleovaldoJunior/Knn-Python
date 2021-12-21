import cv2
import numpy as np

img = cv2.imread('imagens/GorisRaioni.jpg')

# Trying 4 gamma values.
for gamma in [8]:
    # Apply gamma correction.
    gamma_corrected = np.array(255 * (img / 255) ** gamma, dtype='uint8')

    # Save edited images.
    cv2.imshow('gamma_transformed' + str(gamma) + '.jpg', gamma_corrected)
    cv2.waitKey(0)

img = cv2.imread('imagens/Cerebro.png', 0)

lst = []
for i in range(img.shape[0]):
    for j in range(img.shape[1]):
         lst.append(np.binary_repr(img[i][j] ,width=8)) # width = no. of bits

img_128 = (np.array([int(i[0]) for i in lst],dtype = np.uint8) * 128).reshape(img.shape[0],img.shape[1])
img_64 = (np.array([int(i[1]) for i in lst],dtype = np.uint8) * 64).reshape(img.shape[0],img.shape[1])
img_32 = (np.array([int(i[2]) for i in lst],dtype = np.uint8) * 32).reshape(img.shape[0],img.shape[1])
img_final = img_128+img_64
cv2.imshow("Imagem top",img_final)
cv2.waitKey(0)

img = cv2.imread('imagens/Celula.png',0)

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

t1 = transformacao(img)

cv2.imshow('t1', t1)
cv2.waitKey(0)

img = cv2.imread('imagens/Floresta.png')

for gamma in [.9]:
    gamma_corrected = np.array(255 * (img / 255) ** gamma, dtype='uint8')

    cv2.imshow('gamma_transformed' + str(gamma) + '.jpg', gamma_corrected)
    cv2.waitKey(0)
