import cv2
import numpy as np

img = cv2.imread('imagens/Fig0309(a)(washed_out_aerial_image).png')

for gamma in [3,4,5]:
    #Aplico a correção gamma
    gamma = np.array(255 * (img / 255) ** gamma, dtype='uint8')
    cv2.imshow('gamma_transformed' + str(gamma) + '.jpg', gamma)
    cv2.waitKey(0)

