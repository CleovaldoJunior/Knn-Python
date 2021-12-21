import cv2 as cv
import numpy as np


intensidades = [0, 64, 192]

for i in intensidades:
    quadrado = np.zeros((300,300), np.uint8)
    cv.rectangle(quadrado, (0,0), (300,300), i, -1)
    cv.rectangle(quadrado, (100,100), (200,200), 128, -1)
    cv.imshow('Quadrado de intensidade '+str(i),quadrado)
    cv.waitKey(0)