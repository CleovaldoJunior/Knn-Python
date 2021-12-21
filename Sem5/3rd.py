import cv2
import numpy as np

img = cv2.imread('imagens/Fig0308(a)(fractured_spine).png')

v = [.6, .5, .4]

for g in v:
    #Aplico a correção gamma
    c = 255 / (np.log(1 + np.max(img)*g))
    log = c * np.log(1 + img*g)

    log = np.array(log, dtype=np.uint8)

    cv2.imshow(str(g),log)
    cv2.waitKey(0)