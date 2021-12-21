import cv2
import numpy as np

img = cv2.imread('imagens/a4d88a27b6e6f33558a8e675b742-1458995.jpg')

for gamma in [.3,.4,.5,.6,.7,.8]:
    #Aplico a correção gamma
    gamma_corrected = np.array(255 * (img / 255) ** gamma, dtype='uint8')

    cv2.imshow('gamma_transformed' + str(gamma) + '.jpg', gamma_corrected)
    cv2.waitKey(0)
