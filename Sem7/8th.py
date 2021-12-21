import cv2

img = cv2.imread('imagens/Fig0335(a)(ckt_board_saltpep_prob_pt05).png',0)
img_filtro_3 = cv2.blur(img,(3,3))
cv2.imshow('media_3', img_filtro_3)
cv2.waitKey(0)
img_median_3 = cv2.medianBlur(img,3)
cv2.imshow('median_3', img_median_3)
cv2.waitKey(0)