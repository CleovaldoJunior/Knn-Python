import cv2
import numpy as np
from matplotlib import pyplot as plt

imagens = ["imagens/Fig0323(a)(mars_moon_phobos).png"]

def histograma(img):
    #Pega as intensidades únicas
    intensidades_unicas = list(set(img.flatten()))
    nk_vet = []
    for intensidade in intensidades_unicas:
        img.flatten()
        nk = np.count_nonzero(img == intensidade)
        nk_vet.append(nk / (img.shape[0] * img.shape[1]))
    plt.bar(intensidades_unicas, nk_vet, align='center')
    plt.ylim((0, 0.12))
    plt.ylabel('Prk')
    plt.xlabel(str("Intensidade"))
    plt.show()

def letraA(imagens):
    for imagem in imagens:
        img = cv2.imread(imagem,0)
        intensidades_unicas = list(set(img.flatten()))
        topi = []
        cv2.imshow(imagem, img)
        cv2.waitKey(0)
        for intensidade in intensidades_unicas:
            img.flatten()
            nk = np.count_nonzero(img == intensidade)
            topi.append(nk/(img.shape[0]*img.shape[1]))

        plt.bar(intensidades_unicas,topi, align='center')
        plt.xlim((0,255))
        if imagem == "imagens/Fig0323(a)(mars_moon_phobos).png" or imagem == "imagens/Fig0308(a)(fractured_spine).png":
            plt.ylim((0,0.0))
        plt.ylabel('Prk')
        plt.xlabel(str("Intesidade"))
        plt.show()


def equalizar(imagens):
    for imagem in imagens:
        img = cv2.imread(imagem)
        img_to_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        img_to_yuv[:, :, 0] = cv2.equalizeHist(img_to_yuv[:, :, 0])
        hist_equalization_result = cv2.cvtColor(img_to_yuv, cv2.COLOR_YUV2BGR)
        histograma(hist_equalization_result)
        cv2.imshow('result.jpg', hist_equalization_result)
        cv2.waitKey(0)

equalizar(imagens=imagens)

