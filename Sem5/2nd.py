import cv2
import numpy as np

img = cv2.imread('imagens/Fig0308(a)(fractured_spine).png')

# Trying 4 gamma values.
for gamma in [.6,.5,.4]:
    #Aplico a correção de gamma
    gamma_corrected = np.array(255 * (img / 255) ** gamma, dtype='uint8')
    cv2.imshow('gamma_transformed' + str(gamma) + '.jpg', gamma_corrected)
    cv2.waitKey(0)
