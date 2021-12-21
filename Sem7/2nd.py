import cv2
import numpy as np
from matplotlib import pyplot as plt

def histograma(img):
    intensidades_unicas = list(set(img.flatten()))
    nk_vet = []
    for intensidade in intensidades_unicas:
        img.flatten()
        nk = np.count_nonzero(img == intensidade)
        nk_vet.append(nk / (img.shape[0] * img.shape[1]))

    plt.bar(intensidades_unicas, nk_vet, align='center')
    plt.ylim((0))
    plt.ylabel('Prk')
    plt.xlabel(str("Intesidade de pixel"))
    plt.show()

def limiar(imagem, delta=0.0):
    global grupo1_w
    img = imagem.flatten()
    intensidades_unicas = np.unique(img)
    limiar = max(intensidades_unicas)/2
    grupo1 = intensidades_unicas[intensidades_unicas > limiar]
    grupo2 = intensidades_unicas[intensidades_unicas <= limiar]
    novo_limiar = (np.median(grupo1) + np.median(grupo2)) / 2
    while(abs(limiar - novo_limiar) > delta):
        grupo1_w = intensidades_unicas[intensidades_unicas > novo_limiar]
        grupo2_w = intensidades_unicas[intensidades_unicas <= novo_limiar]
        limiar = novo_limiar
        novo_limiar = (np.median(grupo1_w)+np.median(grupo2_w))/2
    return limiar

def otsu(img):
    histograma, limite = np.histogram(img, bins=256)
    limites = (limite[:-1] + limite[1:]) / 2.
    p1k = np.cumsum(histograma)
    p2k = np.cumsum(histograma[::-1])[::-1]
    m1k = np.cumsum((histograma*limites)) / p1k
    m2k = (np.cumsum((histograma*limites)[::-1]) / p2k[::-1])[::-1]
    delta_classes = p1k[:-1] * p2k[1:] * (m1k[:-1] - m2k[1:]) ** 2
    limiar_retorno = limites[:-1][np.argmax(delta_classes)]
    _, img_limiar = cv2.threshold(img, limiar_retorno, 255, cv2.THRESH_BINARY)
    cv2.imwrite('Imagem_limiar_otsu.png', img_limiar)


img = cv2.imread("imagens/Fig1039(a)(polymersomes).png", 0)

histograma(img)
otsu(img)
limiar_retorno = limiar(img, 0.5)
_, img_limiar = cv2.threshold(img, limiar_retorno, 255, cv2.THRESH_BINARY)
cv2.imwrite('Imagem_limiar.png', img_limiar)
