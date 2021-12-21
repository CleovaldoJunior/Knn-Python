import cv2 as cv

lena = cv.imread('imagens/lena_gray_512.png', cv.IMREAD_GRAYSCALE)
gray = cv.imread('imagens/mandril_gray.png', cv.IMREAD_GRAYSCALE)

k = .75
for i in range(lena.shape[0]):
    for j in range(gray.shape[1]):
        lena[i][j] = k*lena[i][j]+((1-k)*gray[i][j])
cv.imshow("img", lena)
cv.waitKey(0)
