import cv2
import numpy as np

filter = [[[-1, 0], [0, 1]],[[0, -1],[1, 0]]]
img = cv2.imread('imagens/Fig0340(a)(dipxe_text).png',)
sobelX = cv2.filter2D(img, -1, kernel=np.array(filter[0]))
sobelY = cv2.filter2D(img, -1, kernel=np.array(filter[1]))
absx = cv2.convertScaleAbs(sobelX)
absy = cv2.convertScaleAbs(sobelY)
img_final = cv2.addWeighted(absx, 0.5, absy, 0.5, 0)
cv2.imshow("Imagem Final",img_final)
cv2.waitKey(0)

