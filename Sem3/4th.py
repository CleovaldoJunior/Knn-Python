import cv2 as cv
import numpy as np

xray = cv.imread('imagens/Fig0230(a)(dental_xray).png',cv.IMREAD_GRAYSCALE)
mask = cv.imread('imagens/Fig0230(b)(dental_xray_mask).png',cv.IMREAD_GRAYSCALE)

def cor(img):
    maximo = np.amax(img)
    minimo = np.min(img)
    return (255*(img-minimo))/(maximo-minimo)

img_top = xray*mask
corrigida = cor(img_top)
cv.imshow("img", corrigida)
cv.waitKey(0)

