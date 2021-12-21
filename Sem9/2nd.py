import cv2

img = cv2.imread('imagens/Fig0340(a)(dipxe_text).png',)

for k in range(1,7):
    imagem_media = cv2.GaussianBlur(img, (5,5), cv2.BORDER_DEFAULT)
    img_sub = img - imagem_media
    saida = img + (k*img_sub)
    cv2.imshow(str(k), saida)
    cv2.waitKey(0)