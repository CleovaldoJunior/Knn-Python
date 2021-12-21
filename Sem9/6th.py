import cv2
import numpy as np

filter = [[[-1,0,1], [-2,0,2],[-1,0,1]],[[1,2,1],[0,0,0],[-1,-2,-1]]]
img = cv2.imread('imagens/Fig1016(a)(building_original).png')

sobelX = cv2.filter2D(img, -1, kernel=np.array(filter[0]))
sobelY = cv2.filter2D(img, -1, kernel=np.array(filter[1]))
absx = cv2.convertScaleAbs(sobelX)
absy = cv2.convertScaleAbs(sobelY)
img_gradient = cv2.addWeighted(absx, 0.5, absy, 0.5, 0)
cv2.imshow("Imagem gradient",img_gradient)
cv2.resizeWindow("Imagem gradient", 600,600)
cv2.waitKey(0)

img_media = cv2.blur(img,(5,5))
sobelX = cv2.filter2D(img_media, -1, kernel=np.array(filter[0]))
sobelY = cv2.filter2D(img_media, -1, kernel=np.array(filter[1]))
absx = cv2.convertScaleAbs(sobelX)
absy = cv2.convertScaleAbs(sobelY)
img_media_gradient = cv2.addWeighted(absx, 0.5, absy, 0.5, 0)
cv2.imshow("Imagem media gradient",img_media_gradient)
cv2.resizeWindow("Imagem media gradient", 600,600)
cv2.waitKey(0)

_, img_gradient_t = cv2.threshold(img_gradient, 80, 255, cv2.THRESH_BINARY)
_, img_media_gradient_t = cv2.threshold(img_media_gradient, 80, 255, cv2.THRESH_BINARY)
cv2.imshow("Imagem gradient Limiar",img_gradient_t)
cv2.resizeWindow("Imagem gradient Limiar", 600,600)
cv2.waitKey(0)
cv2.imshow("Imagem media gradient Limiar", img_media_gradient_t)
cv2.resizeWindow("Imagem media gradient Limiar", 600,600)
cv2.waitKey(0)

