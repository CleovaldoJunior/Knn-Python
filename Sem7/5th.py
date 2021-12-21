import cv2
import cv2 as cv
import numpy as np

def otsu(img,nome):
    histograma, limite = np.histogram(img, bins=256)
    limites = (limite[:-1] + limite[1:]) / 2.
    p1k = np.cumsum(histograma)
    p2k = np.cumsum(histograma[::-1])[::-1]
    m1k = np.cumsum((histograma*limites)) / p1k
    m2k = (np.cumsum((histograma*limites)[::-1]) / p2k[::-1])[::-1]
    delta_classes = p1k[:-1] * p2k[1:] * (m1k[:-1] - m2k[1:]) ** 2
    limiar_retorno = limites[:-1][np.argmax(delta_classes)]
    _, img_limiar = cv2.threshold(img, limiar_retorno, 255, cv2.THRESH_BINARY)
    cv2.imshow(nome+".png", img_limiar)
    cv2.waitKey(0)

img = cv.imread('imagens/Fig1036(c)(gaussian_noise_mean_0_std_50_added).png',0)
otsu(img,"imagem_otsu_sem_processamento")
img_filtro = cv.blur(img,(5,5))
otsu(img_filtro,"imagem_otsu_com_processamento")