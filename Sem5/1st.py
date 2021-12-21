import cv2

img = cv2.imread("imagens/Fig0304(a)(breast_digital_Xray).png")

#Apenas subtraio 255 de cada bit da matriz, gerando seu inverso
img_neg = 255 - img

cv2.imshow("out.png", img_neg)
cv2.waitKey(0)
