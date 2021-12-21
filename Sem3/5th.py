import cv2 as cv
import numpy as np

lena = cv.imread('ruido/lena1.png',cv.IMREAD_GRAYSCALE)

def cor(img):
    maximo = np.amax(img)
    minimo = np.min(img)
    return (255*(img-minimo))/(maximo-minimo)

print(lena)
for i in range(2,101):
    new = cv.imread('ruido/lena'+str(i)+'.png',cv.IMREAD_GRAYSCALE)
    lena = cv.add(new,lena)

corrigida = cor(lena)
cv.imshow("img", lena)
cv.waitKey(0)

