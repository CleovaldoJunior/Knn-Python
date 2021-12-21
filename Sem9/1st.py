import cv2
import numpy as np

filters = [[[0, -1, 0], [-1, 4, -1], [0, -1, 0]],
           [[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]]
for f in filters:
    img = cv2.imread('imagens/Fig0340(a)(dipxe_text).png',)
    l_kernel = np.array(f)
    img_laplace = cv2.filter2D(img, -1, l_kernel)
    cv2.imshow('temp', img_laplace)
    cv2.waitKey(0)